// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/kda/problem.hpp"

namespace ftrain {
namespace {

using Dims = std::vector<std::int64_t>;

const Dims& dimsOf(const std::vector<Tensor>& ports, std::size_t index) {
    return ports[index].getStorageView().getDims();
}

bool portHasMemory(const std::vector<Tensor>& ports, std::size_t index) {
    return ports[index].getStorageView().getMemory() != nullptr;
}

void requireDims(const std::vector<Tensor>& ports, std::size_t index, const Dims& want, const char* name) {
    if (dimsOf(ports, index) != want) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA port %s has the wrong shape", name);
    }
}

void requireDevice(const std::vector<Tensor>& ports, std::initializer_list<std::size_t> indices) {
    for (const std::size_t index : indices) {
        if (ports[index].getStorageView().isHostMemory()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA requires every port in device memory");
        }
    }
}

std::int64_t product(const Dims& dims) {
    std::int64_t n = 1;
    for (const std::int64_t d : dims) { n *= d; }
    return n;
}

// The optional recurrent-state ports: h absent or exactly kdaSavedHElements elements, state
// absent or exactly kdaStateElements elements (any rank); at most one of them is given.
void requireStates(const std::vector<Tensor>& ports, std::size_t h, std::size_t state, const KdaShape& shape,
                   bool has_cu) {
    if (portHasMemory(ports, h) && product(dimsOf(ports, h)) != kdaSavedHElements(shape, has_cu)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA h must hold ftrainKdaSavedHElements elements");
    }
    if (portHasMemory(ports, state) && product(dimsOf(ports, state)) != kdaStateElements(shape, has_cu)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA state must hold ftrainKdaStateElements elements");
    }
    if (portHasMemory(ports, h) && portHasMemory(ports, state)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA h and state are exclusive");
    }
}

// The optional saved qn / kn: both or neither, only with l2norm_qk, shaped [T, H, K].
void requireQkn(const std::vector<Tensor>& ports, std::size_t qn, std::size_t kn, const Dims& qd,
                const KdaAttributes& attributes) {
    const bool has_qn = portHasMemory(ports, qn);
    if (has_qn != portHasMemory(ports, kn)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA qn and kn are given together or not at all");
    }
    if (!has_qn) { return; }
    if (!attributes.isL2NormQk()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA qn and kn require l2norm_qk");
    }
    requireDims(ports, qn, qd, "qn");
    requireDims(ports, kn, qd, "kn");
}

// The checks the forward and backward share: the THD q/k/v/g/beta block, the gate
// parameters, cu_seqlens, and the saved tensors. q / k name the ports that carry the [T, H, K]
// queries and keys (the backward passes qn / kn when it takes no q / k). Returns the shape.
KdaShape validateCommon(const std::vector<Tensor>& ports, std::size_t q, std::size_t k, std::size_t v, std::size_t g,
                        std::size_t beta, std::size_t a_log, std::size_t dt_bias, std::size_t cu, std::size_t q_rstd,
                        std::size_t k_rstd, std::size_t aqk, std::size_t akk, const KdaAttributes& attributes) {
    const Dims& qd = dimsOf(ports, q);
    if (qd.size() != 3) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA q must be rank three [T, H, K]"); }
    const Dims& vd = dimsOf(ports, v);
    if (vd.size() != 3 || vd[0] != qd[0] || vd[1] != qd[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA v must be [T, H, V] with q's T and H");
    }
    KdaShape shape{qd[0], qd[1], qd[2], vd[2], 1};
    requireDims(ports, k, qd, "k");
    requireDims(ports, g, qd, "g");
    requireDims(ports, beta, Dims{shape.tokens, shape.heads}, "beta");
    requireDims(ports, aqk, Dims{shape.tokens, shape.heads, 64}, "aqk");
    requireDims(ports, akk, Dims{shape.tokens, shape.heads, 64}, "akk");
    if (attributes.isGateInKernel()) {
        if (!portHasMemory(ports, a_log) || !portHasMemory(ports, dt_bias)) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA gate_in_kernel requires a_log and dt_bias");
        }
        requireDims(ports, a_log, Dims{shape.heads}, "a_log");
        if (product(dimsOf(ports, dt_bias)) != shape.heads * shape.key_dim) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA dt_bias must hold H * K elements");
        }
    }
    if (portHasMemory(ports, cu)) {
        const Dims& cd = dimsOf(ports, cu);
        if (cd.size() != 1 || cd[0] < 2) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA cu_seqlens must be rank one [N + 1] with N >= 1");
        }
        shape.seqs = cd[0] - 1;
    }
    if (attributes.isL2NormQk()) {
        if (!portHasMemory(ports, q_rstd) || !portHasMemory(ports, k_rstd)) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA l2norm_qk requires q_rstd and k_rstd");
        }
        requireDims(ports, q_rstd, Dims{shape.tokens, shape.heads}, "q_rstd");
        requireDims(ports, k_rstd, Dims{shape.tokens, shape.heads}, "k_rstd");
    }
    return shape;
}

void appendKey(std::vector<std::uint64_t>& key, std::uint64_t& field, const Tensor& tensor) {
    const StorageView& view = tensor.getStorageView();
    const bool has_memory   = view.getMemory() != nullptr;
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(view.getNumericType()));
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(view.getIndexType()));
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(has_memory));
    key.push_back(++field);  // 16-byte alignment class: the kernels use 16 B vector accesses
    key.push_back(has_memory ? static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(view.getMemory()) % 16) : 0);
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(view.getDims().size()));
    for (const std::int64_t dim : view.getDims()) { key.push_back(static_cast<std::uint64_t>(dim)); }
}

std::vector<std::uint64_t> makeKey(const std::vector<Tensor>& ports, const KdaAttributes& attributes) {
    std::vector<std::uint64_t> key;
    key.reserve(ports.size() * 12 + 8);
    std::uint64_t field = 0;
    for (const Tensor& tensor : ports) { appendKey(key, field, tensor); }
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(attributes.isGateInKernel()));
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(attributes.isL2NormQk()));
    return key;
}

}  // namespace

KdaForwardProblem::KdaForwardProblem(std::vector<Tensor> ports, KdaAttributes attributes) : attributes_(attributes) {
    if (ports.size() != kCount) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA forward needs 17 ports"); }
    shape_ = validateCommon(ports, kQ, kK, kV, kG, kBeta, kALog, kDtBias, kCuSeqlens, kQRstd, kKRstd, kAqk, kAkk,
                            attributes_);
    requireDims(ports, kO, dimsOf(ports, kV), "o");
    requireQkn(ports, kQn, kKn, dimsOf(ports, kQ), attributes_);
    requireStates(ports, kH, kState, shape_, portHasMemory(ports, kCuSeqlens));
    requireDevice(ports, {kQ, kK, kV, kG, kBeta, kALog, kDtBias, kCuSeqlens, kO, kQRstd, kKRstd, kAqk, kAkk, kQn, kKn,
                          kH, kState});
    tensors_ = std::move(ports);
}

std::vector<std::uint64_t> KdaForwardProblem::getProblemKey() const { return makeKey(tensors_, attributes_); }

KdaBackwardProblem::KdaBackwardProblem(std::vector<Tensor> ports, KdaAttributes attributes) : attributes_(attributes) {
    if (ports.size() != kCount) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA backward needs 24 ports"); }
    // q / k may be left out when the forward saved qn / kn: the queries and keys are then qn / kn.
    const bool has_qk = portHasMemory(ports, kQ) || portHasMemory(ports, kK);
    if (!has_qk && !portHasMemory(ports, kQn)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA backward needs q and k, or qn and kn");
    }
    const std::size_t q = has_qk ? kQ : kQn, k = has_qk ? kK : kKn;
    shape_ =
        validateCommon(ports, q, k, kV, kG, kBeta, kALog, kDtBias, kCuSeqlens, kQRstd, kKRstd, kAqk, kAkk, attributes_);
    requireQkn(ports, kQn, kKn, dimsOf(ports, q), attributes_);
    requireDims(ports, kDO, dimsOf(ports, kV), "d_o");
    requireDims(ports, kDq, dimsOf(ports, q), "dq");
    requireDims(ports, kDk, dimsOf(ports, k), "dk");
    requireDims(ports, kDv, dimsOf(ports, kV), "dv");
    requireDims(ports, kDg, dimsOf(ports, kG), "dg");
    requireDims(ports, kDBeta, dimsOf(ports, kBeta), "dbeta");
    if (attributes_.isGateInKernel()) {
        if (!portHasMemory(ports, kDALog) || !portHasMemory(ports, kDDtBias)) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "KDA gate_in_kernel backward requires da_log and ddt_bias");
        }
        requireDims(ports, kDALog, dimsOf(ports, kALog), "da_log");
        requireDims(ports, kDDtBias, dimsOf(ports, kDtBias), "ddt_bias");
    }
    requireStates(ports, kH, kState, shape_, portHasMemory(ports, kCuSeqlens));
    requireDevice(ports, {kDO,  kQ,  kK,  kV,  kG,  kBeta, kALog, kDtBias, kCuSeqlens, kQRstd,   kKRstd, kAqk,
                          kAkk, kQn, kKn, kDq, kDk, kDv,   kDg,   kDBeta,  kDALog,     kDDtBias, kH,     kState});
    tensors_ = std::move(ports);
}

std::vector<std::uint64_t> KdaBackwardProblem::getProblemKey() const { return makeKey(tensors_, attributes_); }

}  // namespace ftrain
