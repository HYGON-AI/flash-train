// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/proj_rms_compute_h_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct ProjRmsComputeHFwdRegistrationOrderFinder final {
    static const char* getName() { return "ProjRmsComputeHFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const ProjRmsComputeHFwdProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
