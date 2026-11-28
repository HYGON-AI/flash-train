# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# GDN2 gfx938 device sources, migrated from the linear_atten GDN2 archive
# 05_gfx938_aicc_port.  The architecture list is owned by the top-level
# CMake configuration (FTRAIN_HYGON_HIP_ARCHS); this file only registers the
# sources this architecture actually provides.
ftrain_add_hygon_operator_target(
  ftrain_gdn2_gfx938 gfx938
  bf16/flash_train_gdn2_chunk.hip)
ftrain_add_hygon_operator_target(
  ftrain_gdn2_bwd_gfx938 gfx938
  bf16/flash_train_gdn2_bwd_entry.hip
  bf16/flash_train_gdn2_bwd_dhu.hip
  bf16/flash_train_gdn2_bwd_intra.hip
  bf16/flash_train_gdn2_bwd_parallel.hip
  bf16/flash_train_gdn2_bwd_wy_dqkg.hip)
