// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_CSA_COMPRESSOR_BWD_FINDER_HPP_
#define FTRAIN_FAMILY_CSA_COMPRESSOR_BWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/csa_compressor_bwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registration-order selection: this family has a single implementation.
struct CSACompressorBwdRegistrationOrderFinder final {
    static const char* getName() { return "CSACompressorBwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const CSACompressorBwdProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
