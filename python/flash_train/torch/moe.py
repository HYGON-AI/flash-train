"""Forward-only, single-node MoE operators with raw NN/NT weights."""

import operator
from functools import wraps
import threading

import torch
import torch.distributed as distributed

from ._flash_train_torch import _MoeSession, moe_is_available


def _integer(value, name, minimum=1):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, bool) or not minimum <= result <= 2**31 - 1:
        raise ValueError(f"{name} must be in [{minimum}, 2147483647]")
    return result


def _partition(m_dim, sizes, world_size):
    if sizes is None:
        if m_dim % world_size:
            raise ValueError("uniform M must be divisible by world_size")
        return (m_dim // world_size,) * world_size
    result = tuple(_integer(size, "rank_m_sizes entry") for size in sizes)
    if len(result) != world_size or sum(result) != m_dim:
        raise ValueError("rank_m_sizes must contain world_size positive entries summing to M")
    return result


def _version(tensor, explicit, name):
    if explicit is None:
        try:
            return tensor._version
        except RuntimeError as error:
            raise ValueError(f"{name}_version is required for inference tensors") from error
    try:
        value = operator.index(explicit)
    except TypeError as error:
        raise ValueError(f"{name}_version must be a nonnegative integer") from error
    if isinstance(explicit, bool) or not 0 <= value < 2**64:
        raise ValueError(f"{name}_version must be a uint64 integer")
    return value


def _description(tensor, version=0):
    dtype = {torch.bfloat16: 0, torch.int32: 1, torch.int64: 2}[tensor.dtype]
    return tensor.data_ptr(), list(tensor.shape), dtype, tensor._cdata, version


def _serialized(function):
    @wraps(function)
    def call(self, *arguments, **keywords):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent access to one MoE operator is not supported")
        try:
            return function(self, *arguments, **keywords)
        finally:
            self._lock.release()
    return call


class _ByteAllGather:
    def __init__(self, group, world_size):
        self.group = group
        self.world_size = world_size

    def __call__(self, payload):
        source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        destinations = [torch.empty_like(source) for _ in range(self.world_size)]
        distributed.all_gather(destinations, source, group=self.group)
        return b"".join(bytes(value.tolist()) for value in destinations)


class _MoeOperator:
    def __init__(self, is_ag, m_dim, n_dim, k_dim, num_experts, *, topk, capacity_m,
                 route, rank_m_sizes, process_group, device, n_chunks=0):
        self._lock = threading.Lock()
        self._session = None
        self.m_dim = _integer(m_dim, "m_dim")
        self.n_dim = _integer(n_dim, "n_dim")
        self.k_dim = _integer(k_dim, "k_dim")
        self.num_experts = _integer(num_experts, "num_experts")
        self.topk = _integer(topk, "topk")
        self.capacity_m = _integer(capacity_m, "capacity_m", minimum=0) or self.m_dim
        n_chunks = _integer(n_chunks, "n_chunks", minimum=0)
        if self.capacity_m < self.m_dim or self.topk > self.num_experts:
            raise ValueError("capacity_m must be at least M and topk must not exceed num_experts")
        if not isinstance(route, str):
            raise ValueError("route must be a canonical route name")
        if not moe_is_available():
            raise RuntimeError("this build has no MoE backend (FTRAIN_ENABLE_MOE=OFF)")
        if not distributed.is_available() or not distributed.is_initialized():
            raise RuntimeError("initialize torch.distributed with a Gloo process group before creating MoE operators")
        if distributed.get_backend(process_group) != "gloo":
            raise ValueError("process_group must use Gloo for CPU bootstrap metadata")
        self.rank = distributed.get_rank(process_group)
        self.world_size = distributed.get_world_size(process_group)
        if not 2 <= self.world_size <= 8 or not 0 <= self.rank < self.world_size:
            raise ValueError("MoE requires 2 to 8 participating ranks on one node")
        self.rank_m_sizes = _partition(self.m_dim, rank_m_sizes, self.world_size)
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("device must be a CUDA/HIP device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.route = route
        self._is_ag = is_ag
        self._is_nt = route.startswith("ag_nt_") or route == "rs_nt"
        self._cached_bindings = ()
        with torch.cuda.device(self.device):
            self._session = _MoeSession(
                is_ag, self.rank, self.world_size, self.device.index, _ByteAllGather(process_group, self.world_size),
                self.m_dim, self.n_dim, self.k_dim, self.num_experts, self.topk, self.capacity_m,
                route, list(self.rank_m_sizes), n_chunks, torch.cuda.current_stream(self.device).cuda_stream,
            )

    @property
    def closed(self):
        return self._session is None

    def _require_open(self):
        if self.closed:
            raise RuntimeError("MoE operator is closed")

    def _check_tensor(self, tensor, name, shape, dtypes):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype not in dtypes:
            raise ValueError(f"{name} has an unsupported dtype")
        if tensor.layout != torch.strided or not tensor.is_contiguous() or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be a contiguous tensor with shape {shape}")

    @_serialized
    def _execute(self, input, weight, routing, expert_weight, out, stream, weight_version, routing_version):
        self._require_open()
        local_m = self.rank_m_sizes[self.rank]
        rows = local_m if self._is_ag else self.m_dim * self.topk
        self._check_tensor(input, "input", (rows, self.k_dim), (torch.bfloat16,))
        weight_shape = ((self.num_experts, self.n_dim, self.k_dim) if self._is_nt else
                        (self.num_experts, self.k_dim, self.n_dim))
        self._check_tensor(weight, "weight", weight_shape, (torch.bfloat16,))
        self._check_tensor(routing, "routing", (self.m_dim, self.topk), (torch.int32, torch.int64))
        values = [input, weight, routing]
        if not self._is_ag:
            self._check_tensor(expert_weight, "expert_weight", (self.m_dim, self.topk), (torch.bfloat16,))
            values.append(expert_weight)
        if torch.is_grad_enabled() and any(value.requires_grad for value in values):
            raise RuntimeError("MoE bindings are forward-only; autograd is not implemented (use torch.no_grad())")
        output_shape = (self.m_dim * self.topk if self._is_ag else local_m, self.n_dim)
        if out is not None:
            self._check_tensor(out, "out", output_shape, (torch.bfloat16,))
            if out.requires_grad:
                raise ValueError("out must not require gradients")
            output_begin = out.data_ptr()
            output_end = output_begin + out.numel() * out.element_size()
            for value in values:
                begin = value.data_ptr()
                end = begin + value.numel() * value.element_size()
                if begin < output_end and output_begin < end:
                    raise ValueError("out must not overlap any input tensor")
        weight_version = _version(weight, weight_version, "weight")
        routing_version = _version(routing, routing_version, "routing")
        selected_stream = torch.cuda.current_stream(self.device) if stream is None else stream
        if not isinstance(selected_stream, torch.cuda.Stream) or selected_stream.device != self.device:
            raise ValueError("stream must belong to the operator's device")
        with torch.cuda.device(self.device), torch.cuda.stream(selected_stream):
            if out is None:
                out = torch.empty(output_shape, dtype=torch.bfloat16, device=self.device)
            descriptions = [_description(value) for value in values]
            descriptions[1] = _description(weight, weight_version)
            descriptions[2] = _description(routing, routing_version)
            descriptions.append(_description(out))
            for value in (*values, out):
                value.record_stream(selected_stream)
            self._session.execute(descriptions, selected_stream.cuda_stream)
            torch.autograd.graph.increment_version(out)
        self._cached_bindings = (weight, routing)
        return out

    @_serialized
    def set_rank_m_sizes(self, m_dim, rank_m_sizes=None):
        self._require_open()
        m_dim = _integer(m_dim, "m_dim")
        sizes = _partition(m_dim, rank_m_sizes, self.world_size)
        if m_dim > self.capacity_m:
            raise ValueError("M exceeds the operator's capacity_m")
        with torch.cuda.device(self.device):
            self._session.set_rank_m_sizes(m_dim, list(sizes))
        self.m_dim = m_dim
        self.rank_m_sizes = sizes

    @_serialized
    def synchronize(self):
        self._require_open()
        with torch.cuda.device(self.device):
            self._session.synchronize()

    @_serialized
    def close(self):
        if not self.closed:
            self._session.close()
            self._session = None
            self._cached_bindings = ()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()

    def __del__(self):
        session = getattr(self, "_session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


class AGGroupGemm(_MoeOperator):
    """AG + grouped GEMM; NN weights [E,K,N_local], NT weights [E,N_local,K]."""

    def __init__(self, m_dim, n_dim, k_dim, num_experts, *, topk=1, capacity_m=0,
                 route="ag_nn_ring_chunk", rank_m_sizes=None, process_group=None, device=None):
        super().__init__(True, m_dim, n_dim, k_dim, num_experts, topk=topk, capacity_m=capacity_m,
                         route=route, rank_m_sizes=rank_m_sizes, process_group=process_group, device=device)

    def forward(self, input, weight, routing, *, out=None, stream=None, weight_version=None, routing_version=None):
        return self._execute(input, weight, routing, None, out, stream, weight_version, routing_version)

    __call__ = forward


class MoEReduceRS(_MoeOperator):
    """Grouped GEMM + weighted token reduction + RS; NN [E,K_local,N], NT [E,N,K_local]."""

    def __init__(self, m_dim, n_dim, k_dim, num_experts, *, topk=1, capacity_m=0,
                 route="rs_nn", rank_m_sizes=None, process_group=None, device=None, n_chunks=0):
        super().__init__(False, m_dim, n_dim, k_dim, num_experts, topk=topk, capacity_m=capacity_m,
                         route=route, rank_m_sizes=rank_m_sizes, process_group=process_group, device=device,
                         n_chunks=n_chunks)

    def forward(self, input, weight, routing, expert_weight, *, out=None, stream=None,
                weight_version=None, routing_version=None):
        return self._execute(input, weight, routing, expert_weight, out, stream, weight_version, routing_version)

    __call__ = forward
