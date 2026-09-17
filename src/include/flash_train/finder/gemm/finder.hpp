#ifndef FTRAIN_FINDER_GEMM_FINDER_HPP_
#define FTRAIN_FINDER_GEMM_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/constraints.hpp"

namespace ftrain {

// The Gemm family's placeholder selection policy: it expresses no
// preference -- findCandidates returns no names, and the engine's
// registration-order tail supplies every applicable record. A real Finder
// policy for a family provides the same static functions and returns the
// candidate names ranked best-first; OpsEngine consults the policies in
// pack order and skips names matching no registered record.
template<typename Problem>
struct RegistrationOrderFinder final {
    static const char* getName() { return "RegistrationOrder"; }

    static bool isEnabled(const Problem&, const Constraints&) { return true; }

    static std::vector<std::string> findCandidates(const Problem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
