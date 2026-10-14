// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FLASH_TRAIN_TORCH_HYGON_DISTRIBUTED_BIND_H_
#define FLASH_TRAIN_TORCH_HYGON_DISTRIBUTED_BIND_H_

#include <torch/extension.h>

namespace flash_train_torch {

void bindHygonDistributed(pybind11::module_& module);

}  // namespace flash_train_torch

#endif
