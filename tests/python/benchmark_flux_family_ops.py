# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Formal family benchmark for the optional Hygon distributed backend.

Run one case per torchrun process.  Prefer the file-path form below when an
installed ``tests`` package would otherwise shadow this source namespace:

    FTRAIN_BENCH_IMPL=flash_train FTRAIN_BENCH_AG_ROUTE=native_nk_a2a ...
    torchrun --standalone --nproc-per-node=8 tests/python/benchmark_flux_family_ops.py

FTRAIN_BENCH_IMPL may be ``flash_train`` or ``flux``.  The latter compares against
the upstream Flux direct API with the same tensors and route.

``FTRAIN_BENCH_TOPK`` is the logical routing fanout.  Expanded AG routes pass
one expert id per already-expanded row and therefore use operator topk=1.
MoEReduceRS keeps ``chosen_experts``/``expert_weight`` in ``[M, topk]`` form,
expands only its input rows, and therefore passes the logical topk to its
operator.  Set ``FTRAIN_BENCH_MOE_RAGGED=1`` for a true rank-varying partition,
or provide comma-separated ``FTRAIN_BENCH_RANK_M_SIZES``.
``FTRAIN_BENCH_VERIFY=1`` additionally checks the selected implementation
against a distributed PyTorch reference.
"""

import datetime
import os
import statistics

import torch
import torch.distributed as dist


IMPL = os.getenv("FTRAIN_BENCH_IMPL", "flash_train")
if IMPL == "ftrain":
    IMPL = "flash_train"
MODE = os.getenv("FTRAIN_BENCH_MODE", "ag")
AG_ROUTE = os.getenv("FTRAIN_BENCH_AG_ROUTE", "native_nk_a2a")
MOE_ROUTE = os.getenv("FTRAIN_BENCH_MOE_ROUTE", "a2a_ragged_signal")
MOE_VARIANT = os.getenv("FTRAIN_BENCH_MOE_VARIANT", "nk_direct")
MOE_RAGGED = os.getenv("FTRAIN_BENCH_MOE_RAGGED", "0") == "1"
RANK_M_SIZES_ENV = os.getenv("FTRAIN_BENCH_RANK_M_SIZES")
TOKENS = int(os.getenv("FTRAIN_BENCH_TOKENS", "8192"))
GLOBAL_N = int(os.getenv("FTRAIN_BENCH_GLOBAL_N", "8192"))
K_DIM = int(os.getenv("FTRAIN_BENCH_K", "8192"))
NUM_EXPERTS = int(os.getenv("FTRAIN_BENCH_EXPERTS", "64"))
TOPK = int(os.getenv("FTRAIN_BENCH_TOPK", "6"))
ITERATIONS = int(os.getenv("FTRAIN_BENCH_ITERS", "20"))
WARMUP = int(os.getenv("FTRAIN_BENCH_WARMUP", "3"))
SYNC_EACH = os.getenv("FTRAIN_BENCH_SYNC_EACH", "0") == "1"
VERIFY = os.getenv("FTRAIN_BENCH_VERIFY", "0") == "1"
VERIFY_ATOL = float(os.getenv("FTRAIN_BENCH_VERIFY_ATOL", "2e-2"))
VERIFY_RTOL = float(os.getenv("FTRAIN_BENCH_VERIFY_RTOL", "2e-2"))
SCALE_PATTERN = os.getenv("FTRAIN_BENCH_SCALE_PATTERN", "ones")


def init_group():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=10),
    )
    group = dist.new_group(backend="nccl")
    return group


def all_gather(tensor, group):
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)


def all_gather_ragged_rows(tensor, rank_rows, group):
    max_rows = max(rank_rows)
    padded = torch.zeros(
        (max_rows, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[: tensor.shape[0]].copy_(tensor)
    gathered = [torch.empty_like(padded) for _ in range(len(rank_rows))]
    dist.all_gather(gathered, padded, group=group)
    return torch.cat(
        [gathered[peer][: rank_rows[peer]] for peer in range(len(rank_rows))],
        dim=0,
    )


def timedCall(call):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def report(group, rank, world_size, samples, extra=""):
    sample_tensor = torch.tensor(samples, dtype=torch.float64, device="cuda")
    gathered = [torch.empty_like(sample_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, sample_tensor, group=group)
    if rank == 0:
        values_by_iteration = torch.stack(gathered).cpu()
        rank_medians = values_by_iteration.median(dim=1).values.tolist()
        mean_of_rank_means = values_by_iteration.mean(dim=1).mean().item()
        median_max = values_by_iteration.max(dim=0).values.median().item()
        median_mean = torch.median(values_by_iteration.mean(dim=0)).item()
        suffix = f" {extra}" if extra else ""
        print(
            f"PERF impl={IMPL}{suffix} iterations={ITERATIONS} "
            f"median_ms={statistics.median(rank_medians):.3f} "
            f"median_max_ms={median_max:.3f} "
            f"median_mean_ms={median_mean:.3f} "
            f"mean_of_rank_means_ms={mean_of_rank_means:.3f} "
            f"rank_median_ms={[round(value, 3) for value in rank_medians]}",
            flush=True,
        )


def make_ag_case(group, rank, world_size):
    if RANK_M_SIZES_ENV is not None:
        logical_rank_sizes = [
            int(item) for item in RANK_M_SIZES_ENV.split(",")
        ]
        if (
            len(logical_rank_sizes) != world_size
            or sum(logical_rank_sizes) != TOKENS
        ):
            raise ValueError(
                "FTRAIN_BENCH_RANK_M_SIZES must contain one M per rank and "
                "sum to FTRAIN_BENCH_TOKENS"
            )
        if any(size <= 0 for size in logical_rank_sizes):
            raise ValueError("FTRAIN_BENCH_RANK_M_SIZES entries must be positive")
        if AG_ROUTE not in (
            "native_kn_ring",
            "native_kn_ring_readymask",
        ) and len(set(logical_rank_sizes)) != 1:
            raise ValueError(
                "Unequal FTRAIN_BENCH_RANK_M_SIZES is supported only by "
                "native_kn_ring and native_kn_ring_readymask"
            )
    else:
        logical_rank_sizes = [TOKENS // world_size] * world_size
    local_m = logical_rank_sizes[rank]
    n_dim = GLOBAL_N // world_size
    generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    input_ = torch.empty((local_m, K_DIM), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty(
        (NUM_EXPERTS, K_DIM, n_dim), dtype=torch.bfloat16, device="cuda"
    )
    # GEMM performance is routing/layout-sensitive but not value-sensitive.
    # Avoid generating multi-GiB random activations on every A/B process;
    # numerical runs still use bounded random data.
    if VERIFY:
        input_.uniform_(-0.02, 0.02, generator=generator)
        generator.manual_seed(2000 + rank)
        weight.uniform_(-0.02, 0.02, generator=generator)
    else:
        input_.zero_()
        weight.zero_()
    generator.manual_seed(3000 + rank)
    local_topk = torch.randint(
        0,
        NUM_EXPERTS,
        (local_m, TOPK),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )

    expanded_route = AG_ROUTE != "native_nk_a2a"
    if expanded_route:
        expanded_input = (
            input_[:, None, :]
            .expand(-1, TOPK, -1)
            .reshape(-1, K_DIM)
            .contiguous()
        )
        expanded_ids = local_topk.reshape(-1).contiguous()
        expert_order = torch.argsort(expanded_ids, stable=True)
        input_ = expanded_input.index_select(0, expert_order).contiguous()
        local_topk = (
            expanded_ids.index_select(0, expert_order).reshape(-1, 1).contiguous()
        )
        full_m = TOKENS * TOPK
        operator_topk = 1
        rank_m_sizes = (
            [rows * TOPK for rows in logical_rank_sizes]
            if AG_ROUTE in ("native_kn_ring", "native_kn_ring_readymask")
            else []
        )
    else:
        full_m = TOKENS
        operator_topk = TOPK
        rank_m_sizes = [TOKENS // world_size] * world_size

    gathered_rows = (
        [rows * TOPK for rows in logical_rank_sizes]
        if expanded_route
        else logical_rank_sizes
    )
    topk_ids = all_gather_ragged_rows(local_topk, gathered_rows, group)
    output = torch.empty(
        (full_m * operator_topk, n_dim), dtype=torch.bfloat16, device="cuda"
    )
    return group, input_, weight, topk_ids, output, full_m, operator_topk, rank_m_sizes


def ag_torch_reference(group, input_, weight, topk_ids, output):
    full_m = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    k_dim = input_.shape[1]
    n_dim = output.shape[1]
    full_input = all_gather(input_, group)
    reference = torch.zeros_like(output)
    flat_experts = topk_ids.reshape(-1)
    row_ids = torch.arange(full_m * topk, device=input_.device)
    # Compact native-NK output is token-major.  Expanded native-KN routes pass
    # one expert id per already-expanded row, so every output row maps to the
    # same gathered input row.
    token_ids = (
        row_ids
        if topk == 1
        else torch.div(row_ids, topk, rounding_mode="floor")
    )
    for expert in range(NUM_EXPERTS):
        rows = torch.nonzero(flat_experts == expert, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        expert_input = full_input.index_select(
            0, token_ids.index_select(0, rows)
        )
        expert_output = expert_input.matmul(weight[expert])
        reference.index_copy_(0, rows, expert_output)
    return reference


def verify_ag(group, rank, world_size, input_, weight, topk_ids, output):
    reference = ag_torch_reference(group, input_, weight, topk_ids, output)
    torch.cuda.synchronize()
    difference = (output.float() - reference.float()).abs()
    max_diff = difference.max().clone()
    mean_abs = difference.mean().clone()
    allclose = torch.tensor(
        torch.allclose(
            output.float(),
            reference.float(),
            atol=VERIFY_ATOL,
            rtol=VERIFY_RTOL,
        ),
        device="cuda",
        dtype=torch.int32,
    )
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(mean_abs, op=dist.ReduceOp.SUM, group=group)
    mean_abs /= world_size
    dist.all_reduce(allclose, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        print(
            f"VERIFY impl={IMPL} mode=ag route={AG_ROUTE} "
            f"max_diff={max_diff.item():.9g} "
            f"mean_abs={mean_abs.item():.9g} allclose={bool(allclose.item())} "
            f"(atol={VERIFY_ATOL}, rtol={VERIFY_RTOL})",
            flush=True,
        )
    if not bool(allclose.item()):
        raise RuntimeError(
            f"AGGroupGemm verification failed: max_diff={max_diff.item()} "
            f"allclose={bool(allclose.item())}"
        )


def benchmark_ag(group, rank, world_size):
    (
        group,
        input_,
        weight,
        topk_ids,
        output,
        full_m,
        operator_topk,
        rank_m_sizes,
    ) = make_ag_case(group, rank, world_size)

    if IMPL == "flux":
        import flux

        flux.init_flux_shm(group)
        operator = flux.AGGroupGemm(
            etp_group=group,
            nnodes=1,
            full_m=full_m,
            n_dim=output.shape[1],
            k_dim=K_DIM,
            num_experts=NUM_EXPERTS,
            topk=operator_topk,
            input_dtype=torch.bfloat16,
            output_dtype=None,
            allocate_output_on_init=False,
            rank_m_sizes=rank_m_sizes,
            capacity_full_m=-1,
            route=AG_ROUTE,
            variant="",
        )

        def call():
            operator.forward(input_, weight, topk_ids, output)

    else:
        import flash_train.torch as flash_train_torch

        flash_train_torch.init_hygon_shm(group)
        session_id = flash_train_torch.create_ag_group_gemm_session(
            etp_group=group,
            nnodes=1,
            full_m=full_m,
            n_dim=output.shape[1],
            k_dim=K_DIM,
            num_experts=NUM_EXPERTS,
            topk=operator_topk,
            dtype=torch.bfloat16,
            rank_m_sizes=rank_m_sizes,
            capacity_full_m=-1,
            route=AG_ROUTE,
        )
        plan = flash_train_torch.create_ag_group_gemm_plan(
            session_id, input_, weight, topk_ids, output
        )

        def call():
            plan.execute()

    if VERIFY:
        call()
        torch.cuda.synchronize()
        verify_ag(group, rank, world_size, input_, weight, topk_ids, output)

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    dist.barrier(group)
    samples = []
    for _ in range(ITERATIONS):
        if SYNC_EACH:
            dist.barrier(group)
        samples.append(timedCall(call))
    torch.cuda.synchronize()
    report(group, rank, world_size, samples, f"mode=ag route={AG_ROUTE}")


def make_moe_rank_m_sizes(world_size):
    if TOKENS < world_size:
        raise ValueError(
            f"FTRAIN_BENCH_TOKENS={TOKENS} must be at least world_size={world_size}"
        )
    if RANK_M_SIZES_ENV is not None:
        sizes = [int(item) for item in RANK_M_SIZES_ENV.split(",")]
        if len(sizes) != world_size:
            raise ValueError(
                "FTRAIN_BENCH_RANK_M_SIZES must contain one M per rank; "
                f"got {len(sizes)} entries for world_size={world_size}"
            )
        if sum(sizes) != TOKENS:
            raise ValueError(
                f"sum(FTRAIN_BENCH_RANK_M_SIZES)={sum(sizes)} must equal "
                f"FTRAIN_BENCH_TOKENS={TOKENS}"
            )
        if any(size <= 0 for size in sizes):
            raise ValueError("FTRAIN_BENCH_RANK_M_SIZES entries must be positive")
        return sizes

    base, remainder = divmod(TOKENS, world_size)
    sizes = [base + (1 if peer < remainder else 0) for peer in range(world_size)]
    if not MOE_RAGGED or world_size < 2:
        return sizes

    # Keep the historical eight-rank perturbation for reproducibility.  The
    # fallback preserves the global token count while keeping every rank
    # non-empty on other world sizes.
    if world_size == 8:
        deltas = [17, -7, 9, -11, 5, -13, 0, 0]
        max_negative = max(-delta for delta in deltas if delta < 0)
        limit = min(sizes[1], sizes[3], sizes[5]) - 1
        scale = min(1, limit // max_negative) if max_negative > 0 else 1
        if scale > 0:
            return [size + scale * delta for size, delta in zip(sizes, deltas)]
        if sizes[-1] <= 1:
            raise ValueError(
                "auto eight-rank ragged-M is too small; provide "
                "FTRAIN_BENCH_RANK_M_SIZES explicitly"
            )
        sizes[0] += 1
        sizes[-1] -= 1
        return sizes

    half = world_size // 2
    for left in range(half):
        right = world_size - left - 1
        if sizes[right] > 1:
            sizes[left] += 1
            sizes[right] -= 1
    return sizes


def make_moe_case(group, rank, world_size):
    if GLOBAL_N % world_size or K_DIM % world_size:
        raise ValueError("GLOBAL_N and K must be divisible by world_size")
    if MOE_ROUTE == "a2a" and MOE_RAGGED:
        raise ValueError(
            "FTRAIN_BENCH_MOE_RAGGED=1 requires "
            "FTRAIN_BENCH_MOE_ROUTE=a2a_ragged_signal"
        )
    if TOPK <= 0:
        raise ValueError(f"FTRAIN_BENCH_TOPK={TOPK} must be positive")
    if SCALE_PATTERN not in ("ones", "normalized_random"):
        raise ValueError(f"unknown FTRAIN_BENCH_SCALE_PATTERN: {SCALE_PATTERN}")

    n_dim = GLOBAL_N // world_size
    k_dim = K_DIM // world_size
    native_kn = MOE_VARIANT == "kn_direct"
    rank_m_sizes = make_moe_rank_m_sizes(world_size)
    generator = torch.Generator(device="cuda").manual_seed(2000 + rank)
    input_ = torch.empty(
        (TOKENS * TOPK, k_dim), dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.02, 0.02, generator=generator)
    weight_shape = (
        (NUM_EXPERTS, k_dim, n_dim) if native_kn else (NUM_EXPERTS, n_dim, k_dim)
    )
    weight = torch.empty(
        weight_shape, dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.02, 0.02, generator=generator)
    if rank == 0:
        chosen_experts = torch.randint(
            0,
            NUM_EXPERTS,
            (TOKENS, TOPK),
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
    else:
        chosen_experts = torch.empty(
            (TOKENS, TOPK), dtype=torch.int32, device="cuda"
        )
    dist.broadcast(chosen_experts, src=0, group=group)

    if rank == 0:
        if SCALE_PATTERN == "normalized_random":
            probabilities = torch.rand(
                (TOKENS, TOPK),
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            expert_weight = (
                probabilities / probabilities.sum(dim=1, keepdim=True)
            ).bfloat16()
        else:
            expert_weight = torch.ones(
                (TOKENS, TOPK), dtype=torch.bfloat16, device="cuda"
            )
    else:
        expert_weight = torch.empty(
            (TOKENS, TOPK), dtype=torch.bfloat16, device="cuda"
        )
    dist.broadcast(expert_weight, src=0, group=group)

    output = torch.empty(
        (rank_m_sizes[rank], n_dim), dtype=torch.bfloat16, device="cuda"
    )
    return input_, weight, chosen_experts, expert_weight, output, rank_m_sizes


def moe_torch_reference(
    group,
    input_,
    weight,
    chosen_experts,
    expert_weight,
    output,
    rank_m_sizes,
    native_kn,
):
    tokens, topk = chosen_experts.shape
    k_dim = input_.shape[1]
    n_dim = output.shape[1]
    flat_experts = chosen_experts.reshape(-1)
    flat_scale = expert_weight.reshape(-1)
    routed = torch.empty(
        (tokens * topk, n_dim), dtype=torch.float32, device=input_.device
    )
    for expert in range(NUM_EXPERTS):
        rows = torch.nonzero(flat_experts == expert, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        expert_weight_matrix = (
            weight[expert] if native_kn else weight[expert].T
        )
        expert_output = input_.index_select(0, rows).matmul(
            expert_weight_matrix
        )
        expert_output = expert_output.float()
        expert_output *= flat_scale.index_select(0, rows).float().unsqueeze(1)
        routed.index_copy_(0, rows, expert_output)
    partial = routed.reshape(tokens, topk, n_dim).sum(dim=1).to(input_.dtype)
    reference = torch.empty_like(output)
    dist.reduce_scatter(
        reference,
        list(torch.split(partial, rank_m_sizes, dim=0)),
        group=group,
    )
    return reference


def verify_moe(
    group,
    rank,
    world_size,
    input_,
    weight,
    chosen_experts,
    expert_weight,
    output,
    rank_m_sizes,
):
    reference = moe_torch_reference(
        group,
        input_,
        weight,
        chosen_experts,
        expert_weight,
        output,
        rank_m_sizes,
        MOE_VARIANT == "kn_direct",
    )
    torch.cuda.synchronize()
    difference = (output.float() - reference.float()).abs()
    max_diff = difference.max().clone()
    mean_abs = difference.mean().clone()
    allclose = torch.tensor(
        torch.allclose(
            output.float(),
            reference.float(),
            atol=VERIFY_ATOL,
            rtol=VERIFY_RTOL,
        ),
        device="cuda",
        dtype=torch.int32,
    )
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(mean_abs, op=dist.ReduceOp.SUM, group=group)
    mean_abs /= world_size
    dist.all_reduce(allclose, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        print(
            f"VERIFY impl={IMPL} mode=moe route={MOE_ROUTE} "
            f"variant={MOE_VARIANT} topk={TOPK} "
            f"rank_m_sizes={rank_m_sizes} max_diff={max_diff.item():.9g} "
            f"mean_abs={mean_abs.item():.9g} allclose={bool(allclose.item())} "
            f"(atol={VERIFY_ATOL}, rtol={VERIFY_RTOL})",
            flush=True,
        )
    if not bool(allclose.item()):
        raise RuntimeError(
            f"MoEReduceRS verification failed: max_diff={max_diff.item()} "
            f"allclose={bool(allclose.item())}"
        )


def benchmark_moe(group, rank, world_size):
    (
        input_,
        weight,
        chosen_experts,
        expert_weight,
        output,
        rank_m_sizes,
    ) = make_moe_case(group, rank, world_size)

    if IMPL == "flux":
        import flux

        flux.init_flux_shm(group)
        operator = flux.MoEReduceRS(
            group,
            1,
            TOKENS,
            output.shape[1],
            K_DIM // world_size,
            NUM_EXPERTS,
            TOPK,
            torch.bfloat16,
            0,
            MOE_ROUTE,
            MOE_VARIANT,
        )
        operator.set_rank_m_sizes(TOKENS, rank_m_sizes)

        def call():
            operator.run_a2a(
                input_, weight, chosen_experts, expert_weight, output
            )

    else:
        import flash_train.torch as flash_train_torch

        flash_train_torch.init_hygon_shm(group)
        session_id = flash_train_torch.create_moe_reduce_rs_session(
            etp_group=group,
            nnodes=1,
            max_ntokens=TOKENS,
            n_dim=output.shape[1],
            k_dim=K_DIM // world_size,
            num_experts=NUM_EXPERTS,
            topk=TOPK,
            dtype=torch.bfloat16,
            rank_m_sizes=rank_m_sizes,
            n_chunks=0,
            route=MOE_ROUTE,
            gemm_variant=MOE_VARIANT,
        )
        plan = flash_train_torch.create_moe_reduce_rs_plan(
            session_id, input_, weight, chosen_experts, expert_weight, output
        )

        def call():
            plan.execute()

    if VERIFY:
        call()
        torch.cuda.synchronize()
        verify_moe(
            group,
            rank,
            world_size,
            input_,
            weight,
            chosen_experts,
            expert_weight,
            output,
            rank_m_sizes,
        )

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    dist.barrier(group)
    samples = []
    for _ in range(ITERATIONS):
        if SYNC_EACH:
            dist.barrier(group)
        samples.append(timedCall(call))
    torch.cuda.synchronize()
    report(
        group,
        rank,
        world_size,
        samples,
        f"mode=moe route={MOE_ROUTE} variant={MOE_VARIANT} topk={TOPK} "
        f"partition={'ragged' if len(set(rank_m_sizes)) > 1 else 'equal'}",
    )


def main():
    group = init_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if MODE == "ag":
        benchmark_ag(group, rank, world_size)
    elif MODE == "moe":
        benchmark_moe(group, rank, world_size)
    elif MODE == "all":
        benchmark_ag(group, rank, world_size)
        dist.barrier(group)
        benchmark_moe(group, rank, world_size)
    else:
        raise ValueError(f"unknown FTRAIN_BENCH_MODE: {MODE}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
