# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

ftrain_add_hygon_operator_target(
  ftrain_gdn_${FTRAIN_OPERATOR_ARCH} ${FTRAIN_OPERATOR_ARCH}
  common/flash_train_gdn_chunk.hip)
ftrain_add_hygon_operator_target(
  ftrain_gdn_four_${FTRAIN_OPERATOR_ARCH} ${FTRAIN_OPERATOR_ARCH}
  common/flash_train_gdn_chunk_parallel.hip
  common/flash_train_gdn_state_h.hip)
ftrain_add_hygon_operator_target(
  ftrain_gdn_bwd_${FTRAIN_OPERATOR_ARCH} ${FTRAIN_OPERATOR_ARCH}
  bf16/flash_train_gdn_bwd_entry.hip
  bf16/flash_train_gdn_bwd_dhu.hip
  bf16/flash_train_gdn_bwd_dqkwg.hip
  bf16/flash_train_gdn_bwd_parallel.hip
  bf16/flash_train_gdn_bwd_wy.hip
  bf16/state_transpose.hip)
