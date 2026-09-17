// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "flash_train/family/gemm/finder.hpp"

#include <memory>

#include "flash_train/engine.hpp"
#include "flash_train/family/gemm/family.hpp"

namespace ftrain {
namespace {

// The Gemm engine: the family policy plus its selection policies.
using GemmOpsEngine = OpsEngine<GemmFamily, RegistrationOrderFinder>;

}  // namespace

std::shared_ptr<OpsEngineBase> makeGemmOpsEngine() { return std::make_shared<GemmOpsEngine>(); }

}  // namespace ftrain
