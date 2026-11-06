# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Formal flash-train family checks for the Hygon distributed backend."""

import datetime
import os

import pytest
import torch
import torch.distributed as dist

import flash_train.torch as flash_train_torch


requires_distributed = pytest.mark.skipif(
    "RANK" not in os.environ or "WORLD_SIZE" not in os.environ,
    reason="requires torchrun --standalone --nproc-per-node=N",
)


@pytest.fixture(scope="module")
def etp_group():
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
    flash_train_torch.init_hygon_shm(group)
    yield group
    dist.barrier(group)
    dist.destroy_process_group()


def all_gather(tensor, group):
    gathered = [
        torch.empty_like(tensor) for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)


def max_diff_across_ranks(actual, expected):
    value = (actual.float() - expected.float()).abs().max().clone()
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def make_ag_inputs(group, route, rank, world_size):
    global_n = 2048
    logical_m = 512
    k_dim = 512
    global_n = 2048
    n_dim = global_n // world_size
    num_experts = 64
    logical_topk = 6
    local_m = logical_m // world_size

    generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    logical_input = torch.empty(
        (local_m, k_dim), dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.02, 0.02, generator=generator)
    weight = torch.empty(
        (num_experts, k_dim, n_dim), dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.02, 0.02, generator=generator)
    local_topk = torch.randint(
        0,
        num_experts,
        (local_m, logical_topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )

    expanded_route = route != "native_nk_a2a"
    if expanded_route:
        expanded_input = (
            logical_input[:, None, :]
            .expand(-1, logical_topk, -1)
            .reshape(-1, k_dim)
            .contiguous()
        )
        expanded_ids = local_topk.reshape(-1).contiguous()
        expert_order = torch.argsort(expanded_ids, stable=True)
        input_ = expanded_input.index_select(0, expert_order).contiguous()
        local_topk = (
            expanded_ids.index_select(0, expert_order).reshape(-1, 1).contiguous()
        )
        effective_rank_sizes = []
        if route in ("native_kn_ring", "native_kn_ring_readymask"):
            effective_rank_sizes = [local_m * logical_topk] * world_size
        effective_full_m = logical_m * logical_topk
        operator_topk = 1
    else:
        input_ = logical_input
        effective_rank_sizes = [local_m] * world_size
        effective_full_m = logical_m
        operator_topk = logical_topk

    if effective_rank_sizes:
        local_rows = effective_rank_sizes[rank]
        if local_topk.shape[0] != local_rows:
            raise AssertionError(
                f"route={route} local rows {local_topk.shape[0]} != {local_rows}"
            )

    topk_ids = all_gather(local_topk, group)

    return (
        input_,
        weight,
        topk_ids,
        effective_rank_sizes,
        effective_full_m,
        operator_topk,
    )


@requires_distributed
@pytest.mark.parametrize(
    "route",
    (
        "native_nk_a2a",
        "native_kn_a2a",
        "native_kn_ring_fixed",
        "native_kn_ring",
        "native_kn_ring_readymask",
    ),
)
def test_formal_ag_group_gemm(etp_group, route):
    group = etp_group
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    global_n = 2048
    k_dim = 512
    n_dim = global_n // world_size
    num_experts = 64
    input_, weight, topk_ids, rank_sizes, full_m, topk = make_ag_inputs(
        group, route, rank, world_size
    )
    output = torch.empty(
        (full_m * topk, n_dim), dtype=torch.bfloat16, device="cuda"
    )

    session_id = flash_train_torch.create_ag_group_gemm_session(
        etp_group=group,
        nnodes=1,
        full_m=full_m,
        n_dim=n_dim,
        k_dim=k_dim,
        num_experts=num_experts,
        topk=topk,
        dtype=torch.bfloat16,
        rank_m_sizes=rank_sizes,
        capacity_full_m=-1,
        route=route,
    )
    plan = flash_train_torch.create_ag_group_gemm_plan(
        session_id, input_, weight, topk_ids, output
    )
    route_suffix = route.removeprefix("native_")
    primitive_name = "HygonAGGroupGemm" + {
        "nk_a2a": "NativeNkA2a",
        "kn_a2a": "NativeKnA2a",
        "kn_ring_fixed": "NativeKnRingFixed",
        "kn_ring": "NativeKnRing",
        "kn_ring_readymask": "NativeKnRingReadyMask",
    }[route_suffix]
    assert plan.primitive_name() == primitive_name
    plan.execute()
    torch.cuda.synchronize()

    full_input = all_gather(input_, group)
    reference = torch.zeros_like(output)
    flat_experts = topk_ids.reshape(-1).long()
    token_ids = torch.div(
        torch.arange(full_m * topk, device="cuda"), topk, rounding_mode="floor"
    )
    for expert in range(num_experts):
        rows = torch.nonzero(flat_experts == expert, as_tuple=False).flatten()
        if rows.numel():
            expert_input = full_input.index_select(0, token_ids.index_select(0, rows))
            reference.index_copy_(0, rows, expert_input @ weight[expert])

    max_diff = max_diff_across_ranks(output, reference)
    local_ok = torch.allclose(output.float(), reference.float(), atol=2.5e-2, rtol=2e-2)
    global_ok = torch.tensor(1 if local_ok else 0, device="cuda", dtype=torch.int32)
    dist.all_reduce(global_ok, op=dist.ReduceOp.MIN)
    assert bool(global_ok.item()), f"max_diff={max_diff}"

    plan = None
    flash_train_torch.close_ag_group_gemm_session(session_id)


@requires_distributed
@pytest.mark.parametrize(
    ("route", "gemm_variant", "native_kn_weight"),
    (
        ("a2a", "nk_direct", False),
        ("a2a", "kn_direct", True),
        ("a2a_ragged_signal", "nk_direct", False),
        ("a2a_ragged_signal", "kn_direct", True),
    ),
)
def test_formal_moe_reduce_rs(etp_group, route, gemm_variant, native_kn_weight):
    group = etp_group
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tokens = 128
    n_dim = 512
    k_dim = 256
    num_experts = 16
    topk = 1

    generator = torch.Generator(device="cuda").manual_seed(2000 + rank)
    input_ = torch.empty(
        (tokens, k_dim), dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.02, 0.02, generator=generator)
    weight_shape = (num_experts, k_dim, n_dim) if native_kn_weight else (num_experts, n_dim, k_dim)
    weight = torch.empty(
        weight_shape, dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.02, 0.02, generator=generator)
    if rank == 0:
        chosen_experts = torch.randint(
            0,
            num_experts,
            (tokens, topk),
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
    else:
        chosen_experts = torch.empty(
            (tokens, topk), dtype=torch.int32, device="cuda"
        )
    dist.broadcast(chosen_experts, src=0)
    expert_weight = torch.ones(
        (tokens, topk), dtype=torch.bfloat16, device="cuda"
    )
    output = torch.empty(
        (tokens // world_size, n_dim), dtype=torch.bfloat16, device="cuda"
    )

    session_id = flash_train_torch.create_moe_reduce_rs_session(
        etp_group=group,
        nnodes=1,
        max_ntokens=tokens,
        n_dim=n_dim,
        k_dim=k_dim,
        num_experts=num_experts,
        topk=topk,
        dtype=torch.bfloat16,
        rank_m_sizes=[tokens // world_size for _ in range(world_size)],
        n_chunks=0,
        route=route,
        gemm_variant=gemm_variant,
    )
    plan = flash_train_torch.create_moe_reduce_rs_plan(
        session_id, input_, weight, chosen_experts, expert_weight, output
    )
    route_primitive = "HygonMoEReduceRSRaggedSignal" if route == "a2a_ragged_signal" else "HygonMoEReduceRSA2a"
    assert plan.primitive_name() == route_primitive
    plan.execute()
    torch.cuda.synchronize()

    partial = torch.zeros(
        (tokens * topk, n_dim), dtype=torch.float32, device="cuda"
    )
    flat_experts = chosen_experts.reshape(-1).long()
    for expert in range(num_experts):
        rows = torch.nonzero(flat_experts == expert, as_tuple=False).flatten()
        if rows.numel():
            expert_output = input_.index_select(0, rows).float() @ (
                weight[expert].float() if native_kn_weight else weight[expert].T.float()
            )
            partial.index_copy_(0, rows, expert_output)
    reference = torch.empty_like(output)
    dist.reduce_scatter_tensor(reference, partial.bfloat16())

    max_diff = max_diff_across_ranks(output, reference)
    local_ok = torch.allclose(output.float(), reference.float(), atol=2.5e-2, rtol=2e-2)
    global_ok = torch.tensor(1 if local_ok else 0, device="cuda", dtype=torch.int32)
    dist.all_reduce(global_ok, op=dist.ReduceOp.MIN)
    assert bool(global_ok.item()), (
        f"route={route} variant={gemm_variant} max_diff={max_diff}"
    )

    plan = None
    flash_train_torch.close_moe_reduce_rs_session(session_id)
