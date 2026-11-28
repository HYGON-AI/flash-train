# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# GDN gfx938 device sources, migrated from the linear_atten GDN archive
# 13_gfx938_aicc_port.  The architecture list is owned by the top-level
# CMake configuration (FTRAIN_HYGON_HIP_ARCHS); this file only registers the
# sources this architecture actually provides.
#
# The fused forward target mirrors the archive's gdn_chunk build set: the
# chunked entry dispatches its fused / four-kernel / padded routes itself,
# so FlashTrain exposes a single forward candidate for this architecture and
# keeps no separate four-kernel primitive.  The backward stages mirror the
# archive's gdn_bwd build set, plus the FT-side state transpose glue (its
# entry point lives in ftrain::gfx938 to stay link-unique against gfx936).
ftrain_add_hygon_operator_target(
  ftrain_gdn_gfx938 gfx938
  common/flash_train_gdn_chunk.hip
  common/flash_train_gdn_split.hip
  common/flash_train_gdn_chunk_parallel.hip
  common/flash_train_gdn_state_h.hip
  common/flash_train_gdn_bwd_pad.hip)
ftrain_add_hygon_operator_target(
  ftrain_gdn_bwd_gfx938 gfx938
  bf16/flash_train_gdn_bwd_entry.hip
  bf16/flash_train_gdn_bwd_dhu.hip
  bf16/flash_train_gdn_bwd_dqkwg.hip
  bf16/flash_train_gdn_bwd_parallel.hip
  bf16/flash_train_gdn_bwd_wy.hip
  bf16/state_transpose.hip)
