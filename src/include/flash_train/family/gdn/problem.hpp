// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_GDN_PROBLEM_HPP_
#define FTRAIN_FAMILY_GDN_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/operation/gdn.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {

template<OperationKind Kind>
class GdnProblem final {
  public:
    GdnProblem(std::vector<Tensor> ports, GdnAttributes<Kind> attributes);

    std::uint64_t getB() const noexcept { return getDim(0, 0); }
    std::uint64_t getS() const noexcept { return getDim(0, 1); }
    std::uint64_t getH() const noexcept { return getDim(0, 2); }
    std::uint64_t getHV() const noexcept { return getDim(2, 2); }
    std::uint64_t getK() const noexcept { return getDim(0, 3); }
    std::uint64_t getV() const noexcept { return getDim(2, 3); }
    bool isGqa() const noexcept { return getHV() != getH(); }
    int getChunkSize() const noexcept { return 64; }
    float getScale() const noexcept { return attributes_.getScale(); }

    FTrainNumericType getPortNumericType(std::size_t port) const noexcept { return view(port).getNumericType(); }
    FTrainIndexType getPortIndexType(std::size_t port) const noexcept { return view(port).getIndexType(); }
    void* getPortAddress(std::size_t port) const noexcept { return view(port).getMemory(); }
    const std::vector<std::int64_t>& getPortStrides(std::size_t port) const noexcept { return view(port).getStrides(); }
    std::uint32_t getPortAlignment(std::size_t port) const noexcept;
    bool isPortContinuous(std::size_t port) const noexcept;
#define FTRAIN_GDN_PORT_GETTERS(Name, Port, Valid)                                               \
    FTrainNumericType get##Name##NumericType() const noexcept {                                  \
        static_assert(Valid, #Name " is not a port of this operation");                          \
        return getPortNumericType(Port);                                                          \
    }                                                                                             \
    FTrainIndexType get##Name##IndexType() const noexcept {                                       \
        static_assert(Valid, #Name " is not a port of this operation");                          \
        return getPortIndexType(Port);                                                            \
    }                                                                                             \
    void* get##Name##Address() const noexcept {                                                   \
        static_assert(Valid, #Name " is not a port of this operation");                          \
        return getPortAddress(Port);                                                              \
    }                                                                                             \
    const std::vector<std::int64_t>& get##Name##Strides() const noexcept {                         \
        static_assert(Valid, #Name " is not a port of this operation");                          \
        return getPortStrides(Port);                                                              \
    }                                                                                             \
    std::uint32_t get##Name##Alignment() const noexcept {                                         \
        static_assert(Valid, #Name " is not a port of this operation");                          \
        return getPortAlignment(Port);                                                            \
    }                                                                                             \
    bool is##Name##Continuous() const noexcept {                                                  \
        static_assert(Valid, #Name " is not a port of this operation");                          \
        return isPortContinuous(Port);                                                            \
    }
#define FTRAIN_GDN_IS_GDN (Kind == OperationKind::kGdnForward || Kind == OperationKind::kGdnBackward)
#define FTRAIN_GDN_IS_GDN2 (Kind == OperationKind::kGdn2Forward || Kind == OperationKind::kGdn2Backward)
#define FTRAIN_GDN_IS_FORWARD (Kind == OperationKind::kGdnForward || Kind == OperationKind::kGdn2Forward)
#define FTRAIN_GDN_IS_BACKWARD (Kind == OperationKind::kGdnBackward || Kind == OperationKind::kGdn2Backward)
    FTRAIN_GDN_PORT_GETTERS(Q, 0, true)
    FTRAIN_GDN_PORT_GETTERS(K, 1, true)
    FTRAIN_GDN_PORT_GETTERS(V, 2, true)
    FTRAIN_GDN_PORT_GETTERS(G, 3, true)
    FTRAIN_GDN_PORT_GETTERS(Beta, 4, FTRAIN_GDN_IS_GDN)
    FTRAIN_GDN_PORT_GETTERS(BGate, 4, FTRAIN_GDN_IS_GDN2)
    FTRAIN_GDN_PORT_GETTERS(W, 5, FTRAIN_GDN_IS_GDN2)
    FTRAIN_GDN_PORT_GETTERS(H0, Kind == OperationKind::kGdnForward ? 5 : 6, FTRAIN_GDN_IS_GDN)
    FTRAIN_GDN_PORT_GETTERS(Ht, 7, Kind == OperationKind::kGdnForward)
    FTRAIN_GDN_PORT_GETTERS(Out, 6, FTRAIN_GDN_IS_FORWARD)
    FTRAIN_GDN_PORT_GETTERS(Do, FTRAIN_GDN_IS_GDN ? 5 : 6, FTRAIN_GDN_IS_BACKWARD)
    FTRAIN_GDN_PORT_GETTERS(Dht, 7, Kind == OperationKind::kGdnBackward)
    FTRAIN_GDN_PORT_GETTERS(DQ, FTRAIN_GDN_IS_GDN ? 8 : 7, FTRAIN_GDN_IS_BACKWARD)
    FTRAIN_GDN_PORT_GETTERS(DK, FTRAIN_GDN_IS_GDN ? 9 : 8, FTRAIN_GDN_IS_BACKWARD)
    FTRAIN_GDN_PORT_GETTERS(DV, FTRAIN_GDN_IS_GDN ? 10 : 9, FTRAIN_GDN_IS_BACKWARD)
    FTRAIN_GDN_PORT_GETTERS(DG, FTRAIN_GDN_IS_GDN ? 11 : 10, FTRAIN_GDN_IS_BACKWARD)
    FTRAIN_GDN_PORT_GETTERS(DBeta, 12, Kind == OperationKind::kGdnBackward)
    FTRAIN_GDN_PORT_GETTERS(Dh0, 13, Kind == OperationKind::kGdnBackward)
    FTRAIN_GDN_PORT_GETTERS(DBGate, 11, Kind == OperationKind::kGdn2Backward)
    FTRAIN_GDN_PORT_GETTERS(DW, 12, Kind == OperationKind::kGdn2Backward)
    FTrainNumericType getBNumericType() const noexcept { return getBGateNumericType(); }
#undef FTRAIN_GDN_IS_BACKWARD
#undef FTRAIN_GDN_IS_FORWARD
#undef FTRAIN_GDN_IS_GDN2
#undef FTRAIN_GDN_IS_GDN
#undef FTRAIN_GDN_PORT_GETTERS
    bool hasH0() const noexcept {
        if constexpr (Kind == OperationKind::kGdnForward) return present(5);
        if constexpr (Kind == OperationKind::kGdnBackward) return present(6);
        return false;
    }
    bool hasHt() const noexcept {
        if constexpr (Kind == OperationKind::kGdnForward) return present(7);
        return false;
    }
    bool hasDht() const noexcept {
        if constexpr (Kind == OperationKind::kGdnBackward) return present(7);
        return false;
    }
    bool hasDh0() const noexcept {
        if constexpr (Kind == OperationKind::kGdnBackward) return present(13);
        return false;
    }

    // Key encodes every port's shape, dtype, layout, address presence and
    // alignment class; the device/arch is in Constraints. Scale is not a
    // selection condition and raw addresses are never encoded.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    const StorageView& view(std::size_t port) const noexcept { return ports_[port].getStorageView(); }
    std::uint64_t getDim(std::size_t port, std::size_t dim) const noexcept {
        return static_cast<std::uint64_t>(view(port).getDims()[dim]);
    }
    // Any single zero dimension makes the view's element count zero, which is
    // the whole absence contract (see ftrainGdnForward in flash_train.h): the
    // other three dimensions of an absent view are never read here or in
    // checkView's optional path (problem.cpp), so callers constructing an
    // absent placeholder may put any values in them.
    bool present(std::size_t port) const noexcept {
        const auto& dims = view(port).getDims();
        for (auto dim : dims) {
            if (dim == 0) return false;
        }
        return true;
    }

    std::vector<Tensor> ports_;
    GdnAttributes<Kind> attributes_;
};

using GdnForwardProblem = GdnProblem<OperationKind::kGdnForward>;
using GdnBackwardProblem = GdnProblem<OperationKind::kGdnBackward>;
using Gdn2ForwardProblem = GdnProblem<OperationKind::kGdn2Forward>;
using Gdn2BackwardProblem = GdnProblem<OperationKind::kGdn2Backward>;

}  // namespace ftrain

#endif
