"""One Python MoE rank; bootstrap, reference communication and sampling are explicit."""

import datetime
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as distributed

from . import routing_data


def _mix_tensor(value):
    value = value & 0xffffffff
    value = ((value ^ (value >> 16)) * 0x7feb352d) & 0xffffffff
    value = ((value ^ (value >> 15)) * 0x846ca68b) & 0xffffffff
    return value ^ (value >> 16)


def _fill(tensor, rank, seed, rows=None, nt=False):
    flat = tensor.view(-1)
    columns, reduction = (tensor.shape[-2], tensor.shape[-1]) if nt else (tensor.shape[-1], tensor.shape[-2])
    salt = seed - 2026
    for begin in range(0, flat.numel(), 2**20):
        indices = torch.arange(begin, min(begin + 2**20, flat.numel()), device=tensor.device)
        if rows is not None:
            values = rows[indices // tensor.shape[1]] * 65537 + (indices % tensor.shape[1]) * 257 + rank * 1009 + salt
        else:
            within = indices % (columns * reduction)
            column = within // reduction if nt else within % columns
            inner = within % reduction if nt else within // columns
            values = indices // (columns * reduction) * 65537 + column * 1009 + inner * 257 + rank * 131 + 17 + salt
        flat[begin:begin + indices.numel()] = ((_mix_tensor(values) % 63 - 31).float() / 2048).to(torch.bfloat16)


def _fingerprint(tensor):
    raw = tensor.view(torch.uint8).flatten()
    indices = torch.arange(128, device=tensor.device) * (raw.numel() - 1) // 127
    result = 14695981039346656037
    for byte in raw[indices].cpu().tolist():
        result = ((result ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f"{result:016x}"


class MoeCase:
    def __init__(self, config, rank, gpu_group):
        from flash_train.torch import AGGroupGemm

        self.config, self.rank, self.gpu_group = config, rank, gpu_group
        self.workload = config["workload"]
        workload = self.workload
        self.is_ag = config["op"] == "ag_group_gemm"
        self.sizes = workload["rank_m_sizes"]
        self.nt = workload["weight_layout"] == "NT"
        self.total, self.topk = config["shape"][0], workload["topk"]
        self.columns, self.reduction = workload["operator_n"], workload["operator_k"]
        ids, rows = routing_data(config["shape"], workload)
        self.ids_cpu = torch.tensor(ids, dtype=torch.int64)
        offset = sum(self.sizes[:rank]) * self.topk
        local_rows = rows[offset:offset + self.sizes[rank] * self.topk] if self.is_ag else rows
        self.input = torch.empty((len(local_rows), self.reduction), device="cuda", dtype=torch.bfloat16)
        row_tensor = torch.tensor(local_rows, dtype=torch.int64, device="cuda")
        _fill(self.input, rank, workload["seed"], rows=row_tensor)
        weight_shape = ((workload["experts"], self.columns, self.reduction) if self.nt else
                        (workload["experts"], self.reduction, self.columns))
        self.weight = torch.empty(weight_shape, device="cuda", dtype=torch.bfloat16)
        _fill(self.weight, rank, workload["seed"], nt=self.nt)
        self.routing = self.ids_cpu.to(device="cuda", dtype=torch.int32).reshape(
            workload["operator_m"], workload["operator_topk"])
        self.probabilities = ((torch.arange(self.topk, device="cuda") + 1).float() /
                              (self.topk * (self.topk + 1) // 2)).to(
            torch.bfloat16).expand(self.total, self.topk).contiguous()
        self.output = torch.empty((self.total * self.topk if self.is_ag else self.sizes[rank], self.columns),
                                  device="cuda", dtype=torch.bfloat16)
        self.reference = torch.empty_like(self.output)
        self.order = torch.argsort(self.ids_cpu, stable=True).cuda()
        self.inverse = torch.argsort(self.order)
        self.counts = torch.bincount(self.ids_cpu, minlength=workload["experts"]).tolist()
        common = dict(m_dim=workload["operator_m"], n_dim=self.columns, k_dim=self.reduction,
                      num_experts=workload["experts"], topk=workload["operator_topk"], route=workload["route"],
                      rank_m_sizes=[size * self.topk for size in self.sizes] if self.is_ag else self.sizes)
        self.operator = AGGroupGemm(**common)
        if self.is_ag:
            self.gather_send = torch.empty((max(self.sizes) * self.topk, self.reduction), device="cuda", dtype=torch.bfloat16)
            self.gather_recv = [torch.empty_like(self.gather_send) for _ in self.sizes]
        else:
            self.rs_send = torch.empty((len(self.sizes) * max(self.sizes), self.columns), device="cuda", dtype=torch.bfloat16)
            self.rs_recv = torch.empty((max(self.sizes), self.columns), device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        self.fingerprints = [_fingerprint(tensor) for tensor in
                             (self.input, self.weight, self.routing, self.probabilities)]

    def native(self):
        if self.is_ag:
            return self.operator(self.input, self.weight, self.routing, out=self.output)
        return self.operator(self.input, self.weight, self.routing, self.probabilities, out=self.output)

    def baseline(self):
        if self.is_ag:
            if len(set(self.sizes)) == 1:
                send = self.input
            else:
                self.gather_send.zero_()
                self.gather_send[:self.input.size(0)].copy_(self.input)
                send = self.gather_send
            distributed.all_gather(self.gather_recv, send, group=self.gpu_group)
            source = torch.cat([part[:size * self.topk] for part, size in zip(self.gather_recv, self.sizes)])
        else:
            source = self.input
        ordered = source.index_select(0, self.order)
        grouped = torch.empty((source.size(0), self.columns), device="cuda", dtype=torch.bfloat16)
        begin = 0
        for expert, count in enumerate(self.counts):
            if count:
                weight = self.weight[expert].transpose(0, 1) if self.nt else self.weight[expert]
                torch.mm(ordered[begin:begin + count], weight, out=grouped[begin:begin + count])
            begin += count
        if self.is_ag:
            self.reference.copy_(grouped)
        else:
            original = grouped.index_select(0, self.inverse)
            partial = (original.reshape(self.total, self.topk, self.columns) *
                       self.probabilities[:, :, None]).sum(dim=1)
            if len(set(self.sizes)) == 1:
                send = partial
            else:
                self.rs_send.zero_()
                offset = 0
                for peer, size in enumerate(self.sizes):
                    self.rs_send[peer * max(self.sizes):peer * max(self.sizes) + size].copy_(partial[offset:offset + size])
                    offset += size
                send = self.rs_send
            distributed.reduce_scatter_tensor(self.rs_recv, send, group=self.gpu_group)
            self.reference.copy_(self.rs_recv[:self.sizes[self.rank]])
        return self.reference

    def validate(self):
        self.baseline()
        self.output.fill_(float("nan"))
        self.native()
        torch.cuda.synchronize()
        if not torch.isfinite(self.output).all().item():
            raise AssertionError("native output contains nonfinite values")
        indices = torch.arange(128, device="cuda") * (self.output.numel() - 1) // 127
        actual = self.output.index_select(0, self.order) if self.is_ag else self.output
        torch.testing.assert_close(actual.flatten()[indices].float(), self.reference.flatten()[indices].float(),
                                   atol=5e-4, rtol=0.03)


def measure(function, warmup, iterations):
    distributed.barrier()
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    distributed.barrier()
    samples = []
    started, ended = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        started.record()
        function()
        ended.record()
        ended.synchronize()
        samples.append(started.elapsed_time(ended))
    distributed.barrier()
    return {"samples_ms": samples}


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    rank = int(sys.argv[2])
    torch.cuda.set_device(rank)
    timeout = datetime.timedelta(seconds=int(os.environ["FTRAIN_BENCH_TIMEOUT"]))
    distributed.init_process_group("gloo", rank=rank, world_size=config["workload"]["world_size"],
                                   init_method="file://" + os.environ["FTRAIN_BENCH_RENDEZVOUS"] + "/gloo", timeout=timeout)
    gpu_group = distributed.new_group(backend="nccl", timeout=timeout)
    case = None
    try:
        with torch.no_grad():
            case = MoeCase(config, rank, gpu_group)
            case.validate()
            baseline = measure(case.baseline, config["warmup"], config["iters"])
            stats = measure(case.native, config["warmup"], config["iters"])
            case.validate()
            row = {"tier": "convenience-python", "impl": "flash_train.torch." +
                   "AGGroupGemm", "shape": config["shape"],
                   "fingerprint": case.fingerprints, "stats": stats, "baseline": baseline}
            case.operator.synchronize()
            distributed.barrier()
            case.operator.close()
            print(json.dumps({"rows": [row]}), flush=True)
    finally:
        if case is not None and not case.operator.closed:
            case.operator.close()
        distributed.destroy_process_group(gpu_group)
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
