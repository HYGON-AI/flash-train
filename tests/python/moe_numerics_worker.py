from datetime import timedelta
from pathlib import Path
import sys

import torch
import torch.distributed as distributed

import flash_train.torch as ftrain


def check_devices(world_size):
    if not ftrain.moe_is_available() or torch.cuda.device_count() < world_size:
        print(f"SKIP: requires MoE backend and {world_size} visible devices", flush=True)
        return 77
    if not distributed.is_gloo_available():
        print("SKIP: Gloo is unavailable", flush=True)
        return 77
    for device in range(world_size):
        properties = torch.cuda.get_device_properties(device)
        if not getattr(properties, "gcnArchName", "").startswith("gfx936"):
            print(f"SKIP: device {device} is not gfx936", flush=True)
            return 77
        if torch.cuda.mem_get_info(device)[0] < 4 * 1024**3:
            print(f"SKIP: device {device} has less than 4 GiB free", flush=True)
            return 77
    return 0


def weight_values(experts, columns, reduction, factor, is_nt, active=None):
    inner = torch.arange(reduction) if active is None else active
    expert = torch.arange(1, experts + 1, dtype=torch.float32)[:, None, None]
    column = (torch.arange(columns, dtype=torch.float32) % 5 + 1)
    inner = inner.to(torch.float32) % 3 + 1
    if is_nt:
        return (expert * column[None, :, None] * inner[None, None, :] * factor / 64).to(torch.bfloat16)
    return (expert * inner[None, :, None] * column[None, None, :] * factor / 64).to(torch.bfloat16)


def data(sizes, rank, phase, reduction, columns, experts, topk, is_ag, ids_override=None, weight_multiplier=1):
    total = sum(sizes)
    offset = sum(sizes[:rank])
    active = torch.arange(64) * (reduction - 1) // 63
    rows = torch.arange(total)[:, None] + phase
    indices = torch.arange(64)[None, :]
    sign = torch.where((rows + indices) % 3 == 0, -1, 1)
    sparse = (sign * (rows % 5 + 1) * (indices % 7 + 1) / 64).to(torch.bfloat16)
    local_sparse = sparse[offset:offset + sizes[rank]] if is_ag else sparse.repeat_interleave(topk, dim=0)
    input_host = torch.zeros((local_sparse.size(0), reduction), dtype=torch.bfloat16)
    input_host[:, active] = local_sparse
    if is_ag:
        ids = torch.cat([(torch.arange(size) + phase) * experts // (size + phase) for size in sizes])[:, None]
    else:
        ids = (torch.arange(total)[:, None] + torch.arange(topk)[None, :] + phase) % experts
    if ids_override is not None:
        ids = ids_override
    probabilities = ((torch.arange(topk) + 1) / 4).expand(total, topk).to(torch.bfloat16).contiguous()

    def grouped(factor):
        source = sparse if is_ag else sparse.repeat_interleave(topk, dim=0)
        weights = weight_values(experts, columns, reduction, factor * weight_multiplier, False, active=active)
        expected = torch.empty((source.size(0), columns), dtype=torch.float32)
        for expert in range(experts):
            selected = ids.flatten() == expert
            expected[selected] = (source[selected].float() @ weights[expert].float()).to(torch.bfloat16).float()
        return expected

    if is_ag:
        expected = grouped(rank % 3 + 1 + phase)
    else:
        expected = torch.zeros((total, columns), dtype=torch.float32)
        cache = {}
        for peer in range(len(sizes)):
            factor = peer % 3 + 1 + phase
            if factor not in cache:
                cache[factor] = grouped(factor).reshape(total, topk, columns)
            expected += (cache[factor] * probabilities.float()[:, :, None]).sum(dim=1)
        expected = expected.to(torch.bfloat16).float()[offset:offset + sizes[rank]]
    dtype = torch.int64 if phase == 1 else torch.int32
    return input_host.cuda(), ids.to(device="cuda", dtype=dtype), probabilities.cuda(), expected


def expect_error(function, message):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        if message not in str(error):
            raise AssertionError(f"Expected {message!r}, got {error!r}") from error
    else:
        raise AssertionError(f"Expected an error containing {message!r}")


def run(kind, route, rendezvous, rank, world_size):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    distributed.init_process_group("gloo", init_method=(Path(rendezvous) / "gloo").as_uri(), rank=rank,
                                   world_size=world_size, timeout=timedelta(seconds=90))
    is_ag = kind == "ag"
    is_nt = route.startswith("ag_nt_") or route == "rs_nt"
    experts, columns, reduction = 8, 256 if is_ag else 4096, 32768 if route.endswith("32k") else 1024
    topk = 1 if is_ag else 2
    operation_type = ftrain.AGGroupGemm if is_ag else ftrain.MoEReduceRS
    weight_shape = (experts, columns, reduction) if is_nt else (experts, reduction, columns)
    weight = torch.empty(weight_shape, dtype=torch.bfloat16, device="cuda")
    try:
        with operation_type(16 * world_size, columns, reduction, experts, topk=topk, capacity_m=32 * world_size,
                            route=route) as operation:
            for phase in range(3):
                sizes = [16] * world_size
                if phase == 1:
                    sizes = ([24] * world_size if route == "ag_nt_a2a_chunk" else
                             [12 + 2 * peer for peer in range(world_size)])
                operation.set_rank_m_sizes(sum(sizes), sizes)
                weight.copy_(weight_values(experts, columns, reduction, rank % 3 + 1 + phase, is_nt))
                input, routing, probabilities, expected = data(
                    sizes, rank, phase, reduction, columns, experts, topk, is_ag)
                arguments = (input, weight, routing) if is_ag else (input, weight, routing, probabilities)
                output_shape = (sum(sizes) if is_ag else sizes[rank], columns)
                if phase == 0:
                    expect_error(lambda: operation(input.cpu(), *arguments[1:]), "must be on")
                    expect_error(lambda: operation(input.float(), *arguments[1:]), "dtype")
                    expect_error(lambda: operation(input[:, :-1], *arguments[1:]), "shape")
                    noncontiguous = input.t().contiguous().t()
                    expect_error(lambda: operation(noncontiguous, *arguments[1:]), "contiguous")
                    invalid_ids = routing.float()
                    invalid = (input, weight, invalid_ids) if is_ag else (input, weight, invalid_ids, probabilities)
                    expect_error(lambda: operation(*invalid), "dtype")
                    overlap = weight.view(-1)[:output_shape[0] * columns].view(output_shape)
                    expect_error(lambda: operation(*arguments, out=overlap), "overlap")
                    expect_error(lambda: operation(*arguments, weight_version=-1), "uint64")
                    expect_error(lambda: operation.set_rank_m_sizes(33 * world_size), "capacity_m")
                    with torch.enable_grad():
                        requires_grad = input.detach().requires_grad_()
                        expect_error(lambda: operation(requires_grad, *arguments[1:]), "autograd")
                    if route == "ag_nt_a2a_chunk":
                        expect_error(lambda: operation.set_rank_m_sizes(152, [12 + 2 * peer for peer in range(8)]),
                                     "equal rank sizes")
                        assert operation.m_dim == sum(sizes)
                previous = torch.cuda.current_stream()
                for repeat in range(3):
                    selected_stream = torch.cuda.Stream()
                    selected_stream.wait_stream(previous)
                    with torch.cuda.stream(selected_stream):
                        output = None if repeat == 1 else torch.full(output_shape, float("nan"), dtype=torch.bfloat16,
                                                                    device="cuda")
                        result = operation(*arguments, out=output, stream=selected_stream)
                        if output is not None:
                            assert result is output
                    selected_stream.synchronize()
                    actual = result.float().cpu()
                    if not torch.isfinite(actual).all():
                        raise AssertionError("Nonfinite MoE output")
                    torch.testing.assert_close(actual, expected.float(), rtol=0.02, atol=0.002)
                    maximum = (actual - expected).abs().max().item()
                    print(f"CHECK {route} phase={phase} repeat={repeat} max_abs={maximum}", flush=True)
                    previous = selected_stream
                operation.synchronize()
                distributed.barrier()
            torch.cuda.current_stream().wait_stream(previous)
            weight.mul_(2)
            expected *= 2
            result = operation(*arguments)
            operation.synchronize()
            torch.testing.assert_close(result.float().cpu(), expected.float(), rtol=0.02, atol=0.002)
            print(f"CHECK {route} same_M inplace_weight", flush=True)
            if is_ag:
                changed_ids = torch.cat([torch.arange(size) * experts // size for size in sizes])[:, None]
            else:
                changed_ids = (torch.arange(sum(sizes))[:, None] + torch.arange(topk)[None, :] + phase + 1) % experts
            routing.copy_(changed_ids)
            _, _, _, expected = data(sizes, rank, phase, reduction, columns, experts, topk, is_ag,
                                     ids_override=changed_ids, weight_multiplier=2)
            result = operation(*arguments)
            operation.synchronize()
            torch.testing.assert_close(result.float().cpu(), expected.float(), rtol=0.02, atol=0.002)
            print(f"CHECK {route} same_M inplace_routing", flush=True)
            with torch.inference_mode():
                inference_arguments = tuple(value.clone() for value in arguments)
                expect_error(lambda: operation(*inference_arguments), "required for inference tensors")
                result = operation(*inference_arguments, weight_version=101, routing_version=101)
                operation.synchronize()
                torch.testing.assert_close(result.float().cpu(), expected.float(), rtol=0.02, atol=0.002)
                print(f"CHECK {route} inference_mode explicit_versions", flush=True)
            operation.close()
            operation.close()
            expect_error(operation.synchronize, "closed")
            expect_error(lambda: operation(*arguments), "closed")
        distributed.barrier()
        print(f"PASS {route} rank={rank}", flush=True)
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--check-devices":
            sys.exit(check_devices(int(sys.argv[2])))
        if len(sys.argv) != 6:
            raise ValueError("usage: moe_numerics_worker.py ag|rs route rendezvous rank world_size")
        run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr, flush=True)
        raise
