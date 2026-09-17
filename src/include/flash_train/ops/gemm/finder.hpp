#ifndef FTRAIN_OPS_GEMM_FINDER_HPP_
#define FTRAIN_OPS_GEMM_FINDER_HPP_

#include "flash_train/finder.hpp"
#include "flash_train/ops/gemm/problem.hpp"

namespace ftrain {

// The Gemm family's default selection policy: every registered Primitive is
// a candidate, in registration order, with no reordering.
class RegistrationOrderFinder final : public Finder {
  public:
    bool isEnabled(const Args&, const SelectionContext&) const override { return true; }

    PrimitiveList findCandidates(const PrimitiveList& records, const Args&, const SelectionContext&) const override {
        return records;
    }

    void sortCandidates(const Args&, const SelectionContext&, PrimitiveList&) const override {}
};

}  // namespace ftrain

#endif
