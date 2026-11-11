"""MoE workload descriptions; dimensions are global logical (M, K, N)."""

from .. import register


AG_ROUTES = ("ag_nn_a2a", "ag_nn_ring", "ag_nn_ring_chunk", "ag_nt_a2a",
             "ag_nt_a2a_chunk", "ag_nt_ring_chunk", "ag_nt_a2a_chunk_32k", "ag_nt_ring_chunk_32k")


def mix_bits(value):
    value &= 0xffffffff
    value = ((value ^ (value >> 16)) * 0x7feb352d) & 0xffffffff
    value = ((value ^ (value >> 15)) * 0x846ca68b) & 0xffffffff
    return value ^ (value >> 16)


def partitions(total, world_size, partition):
    if total % world_size:
        raise ValueError("global M must be divisible by world size")
    sizes = [total // world_size] * world_size
    if partition == "ragged":
        step = max(1, total // world_size // (4 * world_size))
        sizes = [size + (2 * rank - world_size + 1) * step for rank, size in enumerate(sizes)]
    if min(sizes) <= 0:
        raise ValueError("rank M sizes must be positive")
    return sizes


def describe(op_name, shape, options):
    logical_m, global_k, global_n = shape
    if op_name != "ag_group_gemm":
        raise ValueError(f"unknown operator: {op_name}")
    is_ag = True
    route = options["route"] or "ag_nn_ring_chunk"
    if route not in AG_ROUTES:
        raise ValueError(f"route {route!r} does not belong to {op_name}")
    world_size = options["world_size"]
    experts, topk = options["experts"], options["topk"]
    if not 2 <= world_size <= 8 or experts <= 0 or not 1 <= topk <= experts:
        raise ValueError("MoE requires 2–8 ranks and 1 <= topk <= experts")
    if max(logical_m * (topk if is_ag else 1), global_k, global_n, experts) > 2**31 - 1:
        raise ValueError("MoE operator dimensions must fit int32")
    if global_k % world_size or global_n % world_size:
        raise ValueError("global K and N must be divisible by world size")
    sizes = partitions(logical_m, world_size, options["partition"])
    if route == "ag_nt_a2a_chunk" and (len(set(sizes)) != 1 or sizes[0] * topk < 8):
        raise ValueError("ag_nt_a2a_chunk requires equal expanded rank M >= 8")
    if route.endswith("_32k") and global_k != 32768:
        raise ValueError("the _32k routes require K=32768")
    if is_ag and options["n_chunks"]:
        raise ValueError("n_chunks is an RS-only configuration")
    return {"route": route, "world_size": world_size, "experts": experts, "topk": topk,
            "rank_m_sizes": sizes, "seed": options["seed"],
            "balanced_topk": is_ag and options["balanced_topk"] == "true", "unique_topk": False,
            "input_layout": "rank-expert-expanded" if is_ag else "token-choice-expanded",
            "weight_layout": "NT" if route.startswith("ag_nt_") else "NN",
            "operator_m": logical_m * topk if is_ag else logical_m,
            "operator_k": global_k if is_ag else global_k // world_size,
            "operator_n": global_n // world_size if is_ag else global_n,
            "operator_topk": 1 if is_ag else topk, "n_chunks": options["n_chunks"],
            "generator": "moe-mixbits-bf16-v1", "timing": "rank-max-events-v1",
            "baseline_output_layout": "expert-major" if is_ag else "token-major",
            "native_output_layout": "rank-expert-expanded" if is_ag else "token-major"}


def routing_data(shape, workload):
    logical_m = shape[0]
    topk, experts = workload["topk"], workload["experts"]
    salt = workload["seed"] - 2026
    ids = [index % experts if workload["balanced_topk"] else mix_bits(index + 3000 + salt) % experts
           for index in range(logical_m * topk)]
    if workload["input_layout"] != "rank-expert-expanded":
        return ids, list(range(logical_m * topk))
    ordered, rows, offset = [], [], 0
    for size in workload["rank_m_sizes"]:
        indices = sorted(range(offset * topk, (offset + size) * topk), key=ids.__getitem__)
        ordered.extend(ids[index] for index in indices)
        rows.extend(index // topk for index in indices)
        offset += size
    return ordered, rows


def run_benchmark(*args, **kwargs):
    from .runner import run_moe

    return run_moe(*args, **kwargs)


def workload_label(workload):
    sizes = ",".join(str(size) for size in workload["rank_m_sizes"])
    return (f"{workload['route']} / TP{workload['world_size']} / E{workload['experts']} "
            f"topk{workload['topk']} / M=[{sizes}] / seed={workload['seed']} / "
            f"balanced={workload['balanced_topk']} / unique={workload['unique_topk']} / "
            f"chunks={workload['n_chunks']}")


for name, routes, smoke in (("ag_group_gemm", AG_ROUTES, (128, 1024, 2048)),):
    register({"name": name, "precisions": ["bf16"], "distributed": True,
              "benchmark_runner": run_benchmark, "routes": routes,
              "suites": {"smoke": [smoke], "standard": [(16384, 16384, 16384)],
                         "32k": [(32768, 32768, 32768)]}})
