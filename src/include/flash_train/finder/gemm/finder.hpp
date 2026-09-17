#ifndef FTRAIN_FINDER_GEMM_FINDER_HPP_
#define FTRAIN_FINDER_GEMM_FINDER_HPP_

#include <memory>
#include <vector>

#include "flash_train/ops_args.hpp"
#include "flash_train/problem/gemm/problem.hpp"
#include "flash_train/primitive/base.hpp"

namespace ftrain {

// The Gemm family's default selection policy: every registered Primitive is
// a candidate, in registration order, with no reordering. A Finder policy
// for another family provides the same three static functions; OpsEngine
// consults them in pack order.
template<typename Problem>
struct RegistrationOrderFinder final {
    static const char* getName() { return "RegistrationOrder"; }

    static bool isEnabled(const Args&, const Constraints&) { return true; }

    static std::vector<std::shared_ptr<const Primitive<Problem>>> findCandidates(
        const std::vector<std::shared_ptr<const Primitive<Problem>>>& records, const Args&, const Constraints&) {
        return records;
    }

    static void sortCandidates(const Args&, const Constraints&,
                               std::vector<std::shared_ptr<const Primitive<Problem>>>&) {}
};

}  // namespace ftrain

#endif
