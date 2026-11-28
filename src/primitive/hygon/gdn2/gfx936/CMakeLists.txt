# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

ftrain_add_hygon_operator_target(
  ftrain_gdn2_${FTRAIN_OPERATOR_ARCH} ${FTRAIN_OPERATOR_ARCH}
  bf16/flash_train_gdn2_chunk.hip)
ftrain_add_hygon_operator_target(
  ftrain_gdn2_bwd_${FTRAIN_OPERATOR_ARCH} ${FTRAIN_OPERATOR_ARCH}
  bf16/flash_train_gdn2_bwd_entry.hip
  bf16/flash_train_gdn2_bwd_dhu.hip
  bf16/flash_train_gdn2_bwd_intra.hip
  bf16/flash_train_gdn2_bwd_parallel.hip
  bf16/flash_train_gdn2_bwd_wy_dqkg.hip)
