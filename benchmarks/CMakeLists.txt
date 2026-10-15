# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# 新算子接入：在 ops/ 下新增实现（参照 ops/gemm.cpp）并在此登记源文件。
add_executable(ftrain_bench_c
  csrc/ftrain_bench.cpp
  csrc/ops/compactify.cpp
  csrc/ops/csa_compressor_bwd.cpp
  csrc/ops/csa_compressor_fwd.cpp
  csrc/ops/dense_attn_score.cpp
  csrc/ops/dense_attn_score_thd.cpp
  csrc/ops/dense_indexer_score.cpp
  csrc/ops/dense_indexer_score_thd.cpp
  csrc/ops/gemm.cpp
  csrc/ops/h_aggregate_bwd.cpp
  csrc/ops/h_aggregate_fwd.cpp
  csrc/ops/h_post_bda_bwd.cpp
  csrc/ops/h_post_bda_fwd.cpp
  csrc/ops/indexer_bwd_dense.cpp
  csrc/ops/indexer_bwd_dense_thd.cpp
  csrc/ops/indexer_bwd_sparse.cpp
  csrc/ops/indexer_fwd.cpp
  csrc/ops/indexer_fwd_thd.cpp
  csrc/ops/indexer_topk.cpp
  csrc/ops/indexer_topk_thd.cpp
  csrc/ops/proj_rms_bwd.cpp
  csrc/ops/proj_rms_compute_h_bwd.cpp
  csrc/ops/proj_rms_compute_h_fwd.cpp
  csrc/ops/proj_rms_fwd.cpp
  csrc/ops/sinkhorn_bwd.cpp
  csrc/ops/sinkhorn_fwd.cpp
  csrc/ops/sparse_attn_bwd.cpp
  csrc/ops/sparse_attn_fwd.cpp
  csrc/ops/sparse_attn_fwd_indexer.cpp
  csrc/ops/sparse_attn_target.cpp
  csrc/ops/sparse_indexer_predict.cpp)

target_include_directories(ftrain_bench_c PRIVATE csrc)
target_link_libraries(ftrain_bench_c PRIVATE flash_train_impl)
