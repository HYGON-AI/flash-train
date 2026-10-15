// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_CSA_COMPRESSOR_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_CSA_COMPRESSOR_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/csa_compressor_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registration-order selection: this family has a single implementation.
struct CSACompressorFwdRegistrationOrderFinder final {
    static const char* getName() { return "CSACompressorFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const CSACompressorFwdProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
