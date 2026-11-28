// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/gdn/problem.hpp"

namespace ftrain {
namespace {

using Dims = std::vector<std::int64_t>;

void checkView(const StorageView& view, const Dims& expected, bool optional = false) {
    const Dims& dims = view.getDims();
    if (dims.size() != expected.size() || view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN port rank or device placement is invalid");
    }
    if (optional) {
        for (auto dim : dims) {
            if (dim == 0) return;
        }
    }
    if (dims != expected) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN port shape is inconsistent"); }
}

void checkState(const StorageView& view, const Dims& expected) {
    checkView(view, expected, true);
    const Dims& dims = view.getDims();
    bool present = true;
    for (auto dim : dims) present &= dim != 0;
    if (present && view.getNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN present state must be fp32");
    }
}

std::uint64_t elementBytes(FTrainNumericType type) {
    switch (type) {
        case FTRAIN_NUMERIC_TYPE_FP64:
        case FTRAIN_NUMERIC_TYPE_INT64:
        case FTRAIN_NUMERIC_TYPE_UINT64: return 8;
        case FTRAIN_NUMERIC_TYPE_FP32:
        case FTRAIN_NUMERIC_TYPE_INT32:
        case FTRAIN_NUMERIC_TYPE_UINT32: return 4;
        case FTRAIN_NUMERIC_TYPE_FP16:
        case FTRAIN_NUMERIC_TYPE_BF16:
        case FTRAIN_NUMERIC_TYPE_INT16:
        case FTRAIN_NUMERIC_TYPE_UINT16: return 2;
        default: return 1;
    }
}

void checkSize(const StorageView& view) {
    std::uint64_t count = 1;
    for (auto dim : view.getDims()) {
        if (dim == 0) return;
        if (count > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
            throw Exception(FTRAIN_STATUS_OVERFLOW, "GDN element count overflows uint64_t");
        }
        count *= static_cast<std::uint64_t>(dim);
    }
    if (count > std::numeric_limits<std::uint64_t>::max() / elementBytes(view.getNumericType())) {
        throw Exception(FTRAIN_STATUS_OVERFLOW, "GDN byte count overflows uint64_t");
    }
}

}  // namespace

template<OperationKind Kind>
GdnProblem<Kind>::GdnProblem(std::vector<Tensor> ports, GdnAttributes<Kind> attributes)
    : ports_(std::move(ports)), attributes_(attributes) {
    constexpr std::size_t count = Kind == OperationKind::kGdnForward ? 8
                                : Kind == OperationKind::kGdnBackward ? 14
                                : Kind == OperationKind::kGdn2Forward ? 7 : 13;
    if (ports_.size() != count || !std::isfinite(getScale()) || getScale() <= 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN port count or scale is invalid");
    }
    const Dims& q = view(0).getDims();
    const Dims& v = view(2).getDims();
    if (q.size() != 4 || v.size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN q/v must have rank four");
    }
    for (auto dim : q) if (dim <= 0) throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN q dimensions must be positive");
    for (auto dim : v) if (dim <= 0) throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN v dimensions must be positive");
    const auto B = q[0], S = q[1], H = q[2], K = q[3], HV = v[2], V = v[3];
    if (v[0] != B || v[1] != S || HV % H != 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN batch, sequence or head dimensions disagree");
    }
    const Dims qdims{B, S, H, K}, vdims{B, S, HV, V};
    checkView(view(0), qdims);
    checkView(view(1), qdims);
    checkView(view(2), vdims);
    if constexpr (Kind == OperationKind::kGdnForward || Kind == OperationKind::kGdnBackward) {
        const Dims gates{B, S, HV}, state{B, HV, K, V};
        checkView(view(3), gates);
        checkView(view(4), gates);
        if constexpr (Kind == OperationKind::kGdnForward) {
            checkState(view(5), state);
            checkView(view(6), vdims);
            checkState(view(7), state);
            if (view(6).getNumericType() != view(0).getNumericType()) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN output dtype must match q");
            }
        } else {
            checkView(view(5), vdims);
            checkState(view(6), state);
            checkState(view(7), state);
            checkView(view(8), qdims);
            checkView(view(9), qdims);
            checkView(view(10), vdims);
            checkView(view(11), gates);
            checkView(view(12), gates);
            checkState(view(13), state);
        }
    } else {
        const Dims gates{B, S, HV, K};
        checkView(view(3), gates);
        checkView(view(4), gates);
        checkView(view(5), vdims);
        if constexpr (Kind == OperationKind::kGdn2Forward) {
            checkView(view(6), vdims);
            if (view(6).getNumericType() != view(0).getNumericType()) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN2 output dtype must match q");
            }
        } else {
            checkView(view(6), vdims);
            checkView(view(7), qdims);
            checkView(view(8), qdims);
            checkView(view(9), vdims);
            checkView(view(10), gates);
            checkView(view(11), gates);
            checkView(view(12), vdims);
        }
    }
    for (const Tensor& port : ports_) checkSize(port.getStorageView());
}

template<OperationKind Kind>
std::uint32_t GdnProblem<Kind>::getPortAlignment(std::size_t port) const noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(view(port).getMemory());
    if (address == 0) return 0;
    for (std::uint32_t alignment = 16; alignment > 1; alignment /= 2) {
        if (address % alignment == 0) return alignment;
    }
    return 1;
}

template<OperationKind Kind>
bool GdnProblem<Kind>::isPortContinuous(std::size_t port) const noexcept {
    const StorageView& storage = view(port);
    if (storage.getIndexType() == FTRAIN_INDEX_TYPE_CONTINUOUS) return true;
    if (storage.getIndexType() != FTRAIN_INDEX_TYPE_INVALID) return false;
    std::int64_t stride = 1;
    const auto& dims = storage.getDims();
    const auto& strides = storage.getStrides();
    for (std::size_t i = dims.size(); i-- > 0;) {
        if (strides[i] != stride) return false;
        if (dims[i] != 0 && stride > std::numeric_limits<std::int64_t>::max() / dims[i]) return false;
        stride *= dims[i];
    }
    return true;
}

template<OperationKind Kind>
std::vector<std::uint64_t> GdnProblem<Kind>::getProblemKey() const {
    std::vector<std::uint64_t> key;
    for (std::size_t i = 0; i < ports_.size(); ++i) {
        const StorageView& storage = view(i);
        key.push_back(i);
        key.push_back(storage.getNumericType());
        key.push_back(storage.getIndexType());
        key.push_back(storage.getMemory() != nullptr);
        key.push_back(getPortAlignment(i));
        key.push_back(storage.getDims().size());
        for (auto dim : storage.getDims()) key.push_back(static_cast<std::uint64_t>(dim));
        key.push_back(storage.getStrides().size());
        for (auto stride : storage.getStrides()) key.push_back(static_cast<std::uint64_t>(stride));
    }
    return key;
}

// Explicit instantiation is per out-of-line member (not the whole class):
// the dtype getters above (getBetaNumericType/getBNumericType/getWNumericType)
// are guarded by static_assert for the Kind they are meaningful for, and must
// stay implicitly-instantiated-on-use only; `template class GdnProblem<Kind>;`
// would force every member (including those three) to instantiate for every
// Kind regardless of whether anything calls them, defeating the guard.
template GdnProblem<OperationKind::kGdnForward>::GdnProblem(std::vector<Tensor>,
                                                              GdnAttributes<OperationKind::kGdnForward>);
template std::uint32_t GdnProblem<OperationKind::kGdnForward>::getPortAlignment(std::size_t) const noexcept;
template bool GdnProblem<OperationKind::kGdnForward>::isPortContinuous(std::size_t) const noexcept;
template std::vector<std::uint64_t> GdnProblem<OperationKind::kGdnForward>::getProblemKey() const;

template GdnProblem<OperationKind::kGdnBackward>::GdnProblem(std::vector<Tensor>,
                                                               GdnAttributes<OperationKind::kGdnBackward>);
template std::uint32_t GdnProblem<OperationKind::kGdnBackward>::getPortAlignment(std::size_t) const noexcept;
template bool GdnProblem<OperationKind::kGdnBackward>::isPortContinuous(std::size_t) const noexcept;
template std::vector<std::uint64_t> GdnProblem<OperationKind::kGdnBackward>::getProblemKey() const;

template GdnProblem<OperationKind::kGdn2Forward>::GdnProblem(std::vector<Tensor>,
                                                               GdnAttributes<OperationKind::kGdn2Forward>);
template std::uint32_t GdnProblem<OperationKind::kGdn2Forward>::getPortAlignment(std::size_t) const noexcept;
template bool GdnProblem<OperationKind::kGdn2Forward>::isPortContinuous(std::size_t) const noexcept;
template std::vector<std::uint64_t> GdnProblem<OperationKind::kGdn2Forward>::getProblemKey() const;

template GdnProblem<OperationKind::kGdn2Backward>::GdnProblem(std::vector<Tensor>,
                                                                GdnAttributes<OperationKind::kGdn2Backward>);
template std::uint32_t GdnProblem<OperationKind::kGdn2Backward>::getPortAlignment(std::size_t) const noexcept;
template bool GdnProblem<OperationKind::kGdn2Backward>::isPortContinuous(std::size_t) const noexcept;
template std::vector<std::uint64_t> GdnProblem<OperationKind::kGdn2Backward>::getProblemKey() const;

}  // namespace ftrain
