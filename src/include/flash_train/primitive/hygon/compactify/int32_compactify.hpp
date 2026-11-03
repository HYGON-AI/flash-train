// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_COMPACTIFY_INT32_COMPACTIFY_HPP_
#define FTRAIN_PRIMITIVE_HYGON_COMPACTIFY_INT32_COMPACTIFY_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/compactify/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in int32_compactify.hip; requires a successfully selected/configured primitive.
template<class Implementation>
void launchInt32Compactify(FTrainStream stream, const int* indices, int* output, int* topk_length, int num_rows,
                           int selection_width);

// The Compactify Primitive: the ported row-wise pack, no workspace and no
// dtype chain.
template<class Implementation>
class Int32CompactifyBase : public Primitive<CompactifyProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kSmall = false;
    static constexpr bool kLarge = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const CompactifyProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const CompactifyProblem& problem) override;

  protected:
    Int32CompactifyBase()                           = default;
    Int32CompactifyBase(const Int32CompactifyBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;

    const int* indices_  = nullptr;
    int* output_         = nullptr;
    int* topk_length_    = nullptr;
    int num_rows_        = 0;
    int selection_width_ = 0;
};

class Int32Compactify final : public Int32CompactifyBase<Int32Compactify> {
  public:
    static constexpr const char* kName = "Compactify";
    static constexpr bool kSmall       = true;
};

class Int32CompactifyLarge final : public Int32CompactifyBase<Int32CompactifyLarge> {
  public:
    static constexpr const char* kName = "CompactifyLarge";
    static constexpr bool kLarge       = true;
};

}  // namespace ftrain

#endif
