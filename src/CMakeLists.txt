# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_library(
  flash_train_impl
  OBJECT api/common_api.cpp
         api/gdn_api.cpp
         api/indexer_fwd_api.cpp
         api/indexer_fwd_thd_api.cpp
         api/dense_indexer_score_api.cpp
         api/dense_indexer_score_thd_api.cpp
         api/dense_attn_score_api.cpp
         api/dense_attn_score_thd_api.cpp
         api/sparse_indexer_predict_api.cpp
         api/sparse_attn_target_api.cpp
         api/sparse_attn_fwd_api.cpp
         api/sparse_attn_fwd_indexer_api.cpp
         api/sparse_attn_bwd_api.cpp
         api/indexer_bwd_sparse_api.cpp
         api/indexer_bwd_dense_api.cpp
         api/indexer_bwd_dense_thd_api.cpp
         api/indexer_topk_api.cpp
         api/indexer_topk_thd_api.cpp
         api/compactify_api.cpp
         api/csa_compressor_fwd_api.cpp
         api/csa_compressor_bwd_api.cpp
         api/sinkhorn_fwd_api.cpp
         api/sinkhorn_bwd_api.cpp
         api/h_aggregate_fwd_api.cpp
         api/h_aggregate_bwd_api.cpp
         api/h_post_bda_fwd_api.cpp
         api/h_post_bda_bwd_api.cpp
         api/proj_rms_fwd_api.cpp
         api/proj_rms_bwd_api.cpp
         api/proj_rms_compute_h_fwd_api.cpp
         api/proj_rms_compute_h_bwd_api.cpp
         api/gemm_api.cpp
         api/kda_api.cpp
         api/ops_args_api.cpp
         api/pattern_api.cpp
         api/plan_api.cpp
         ops_args.cpp
         plan.cpp
         constraints.cpp
         error.cpp
         trace.cpp
         storage_view.cpp
         tensor.cpp
         handle.cpp
         cache.cpp
         engine.cpp
         family/indexer_fwd/family.cpp
         family/indexer_fwd/problem.cpp
         family/indexer_fwd_thd/family.cpp
         family/indexer_fwd_thd/problem.cpp
         family/dense_indexer_score/family.cpp
         family/dense_indexer_score/problem.cpp
         family/dense_indexer_score_thd/family.cpp
         family/dense_indexer_score_thd/problem.cpp
         family/sparse_indexer_predict/family.cpp
         family/sparse_indexer_predict/problem.cpp
         family/sparse_attn_fwd/family.cpp
         family/sparse_attn_fwd/problem.cpp
         family/sparse_attn_fwd_indexer/family.cpp
         family/sparse_attn_fwd_indexer/problem.cpp
         family/sparse_attn_bwd/family.cpp
         family/sparse_attn_bwd/problem.cpp
         family/indexer_topk/family.cpp
         family/indexer_topk/problem.cpp
         family/compactify/family.cpp
         family/compactify/problem.cpp
         family/dense_attn_score/family.cpp
         family/dense_attn_score/problem.cpp
         family/dense_attn_score_thd/family.cpp
         family/dense_attn_score_thd/problem.cpp
         family/sparse_attn_target/family.cpp
         family/sparse_attn_target/problem.cpp
         family/indexer_topk_thd/family.cpp
         family/indexer_topk_thd/problem.cpp
         family/indexer_bwd_sparse/family.cpp
         family/indexer_bwd_sparse/problem.cpp
         family/indexer_bwd_dense/family.cpp
         family/indexer_bwd_dense/problem.cpp
         family/indexer_bwd_dense_thd/family.cpp
         family/indexer_bwd_dense_thd/problem.cpp
         family/csa_compressor_fwd/family.cpp
         family/csa_compressor_fwd/problem.cpp
         family/csa_compressor_bwd/family.cpp
         family/csa_compressor_bwd/problem.cpp
         family/sinkhorn_fwd/family.cpp
         family/sinkhorn_fwd/problem.cpp
         family/sinkhorn_bwd/family.cpp
         family/sinkhorn_bwd/problem.cpp
         family/h_aggregate_fwd/family.cpp
         family/h_aggregate_fwd/problem.cpp
         family/h_aggregate_bwd/family.cpp
         family/h_aggregate_bwd/problem.cpp
         family/h_post_bda_fwd/family.cpp
         family/h_post_bda_fwd/problem.cpp
         family/h_post_bda_bwd/family.cpp
         family/h_post_bda_bwd/problem.cpp
         family/proj_rms_fwd/family.cpp
         family/proj_rms_fwd/problem.cpp
         family/proj_rms_bwd/family.cpp
         family/proj_rms_bwd/problem.cpp
         family/proj_rms_compute_h_fwd/family.cpp
         family/proj_rms_compute_h_fwd/problem.cpp
         family/proj_rms_compute_h_bwd/family.cpp
         family/proj_rms_compute_h_bwd/problem.cpp
         family/gemm/family.cpp
         family/gemm/problem.cpp
         family/gdn/family.cpp
         family/gdn/problem.cpp
         family/kda/family.cpp
         family/kda/problem.cpp
         primitive.cpp
         pattern.cpp)
target_include_directories(
  flash_train_impl
  PUBLIC "${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include"
         "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(flash_train_impl PUBLIC Threads::Threads)

# The platform-specific primitives: everything under primitive/hygon/
# builds only for the selected platform and carries its dependencies.
if(FTRAIN_PLATFORM_HYGON_HIP)
  include("${PROJECT_SOURCE_DIR}/cmake/HygonOperatorSources.cmake")
  foreach(FTRAIN_OPERATOR_ARCH IN LISTS FTRAIN_HYGON_HIP_ARCHS)
    foreach(ftrain_operator IN ITEMS gdn gdn2)
      set(ftrain_operator_sources "primitive/hygon/${ftrain_operator}/${FTRAIN_OPERATOR_ARCH}/CMakeLists.txt")
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${ftrain_operator_sources}")
        add_subdirectory("primitive/hygon/${ftrain_operator}/${FTRAIN_OPERATOR_ARCH}")
      else()
        message(STATUS "${ftrain_operator}: no ${FTRAIN_OPERATOR_ARCH} device sources; runtime unsupported")
      endif()
    endforeach()
  endforeach()
  target_sources(
    flash_train_impl
    PRIVATE primitive/hygon/gemm/fp32_gemm.cpp
            primitive/hygon/indexer_fwd/bf16_indexer_fwd.cpp
            primitive/hygon/indexer_fwd/fp32_indexer_fwd.cpp
            primitive/hygon/indexer_fwd/mixed_fp32_indexer_fwd.cpp
            primitive/hygon/indexer_fwd/fp16_indexer_fwd.cpp
            primitive/hygon/indexer_fwd_thd/bf16_indexer_fwd_thd.cpp
            primitive/hygon/indexer_fwd_thd/fp32_indexer_fwd_thd.cpp
            primitive/hygon/indexer_fwd_thd/mixed_fp32_indexer_fwd_thd.cpp
            primitive/hygon/indexer_fwd_thd/fp16_indexer_fwd_thd.cpp
            primitive/hygon/dense_indexer_score/bf16_dense_indexer_score.cpp
            primitive/hygon/dense_indexer_score/fp32_dense_indexer_score.cpp
            primitive/hygon/dense_indexer_score/mixed_fp32_dense_indexer_score.cpp
            primitive/hygon/dense_indexer_score/fp16_dense_indexer_score.cpp
            primitive/hygon/dense_indexer_score_thd/bf16_dense_indexer_score_thd.cpp
            primitive/hygon/dense_indexer_score_thd/fp32_dense_indexer_score_thd.cpp
            primitive/hygon/dense_indexer_score_thd/mixed_fp32_dense_indexer_score_thd.cpp
            primitive/hygon/dense_indexer_score_thd/fp16_dense_indexer_score_thd.cpp
            primitive/hygon/sparse_indexer_predict/bf16_sparse_indexer_predict.cpp
            primitive/hygon/sparse_indexer_predict/fp32_sparse_indexer_predict.cpp
            primitive/hygon/sparse_indexer_predict/mixed_fp32_sparse_indexer_predict.cpp
            primitive/hygon/sparse_indexer_predict/fp16_sparse_indexer_predict.cpp
            primitive/hygon/sparse_attn_fwd/bf16_sparse_attn_fwd.cpp
            primitive/hygon/sparse_attn_fwd/fp32_sparse_attn_fwd.cpp
            primitive/hygon/sparse_attn_fwd/mixed_fp32_sparse_attn_fwd.cpp
            primitive/hygon/sparse_attn_fwd/fp16_sparse_attn_fwd.cpp
            primitive/hygon/sparse_attn_fwd_indexer/bf16_sparse_attn_fwd_indexer.cpp
            primitive/hygon/sparse_attn_fwd_indexer/fp32_sparse_attn_fwd_indexer.cpp
            primitive/hygon/sparse_attn_fwd_indexer/mixed_fp32_sparse_attn_fwd_indexer.cpp
            primitive/hygon/sparse_attn_fwd_indexer/fp16_sparse_attn_fwd_indexer.cpp
            primitive/hygon/sparse_attn_bwd/bf16_sparse_attn_bwd.cpp
            primitive/hygon/sparse_attn_bwd/fp32_sparse_attn_bwd.cpp
            primitive/hygon/sparse_attn_bwd/mixed_fp32_sparse_attn_bwd.cpp
            primitive/hygon/sparse_attn_bwd/fp16_sparse_attn_bwd.cpp
            primitive/hygon/indexer_topk/fp32_indexer_topk.cpp
            primitive/hygon/compactify/int32_compactify.cpp
            primitive/hygon/dense_attn_score/bf16_dense_attn_score.cpp
            primitive/hygon/dense_attn_score/fp32_dense_attn_score.cpp
            primitive/hygon/dense_attn_score/mixed_fp32_dense_attn_score.cpp
            primitive/hygon/dense_attn_score/fp16_dense_attn_score.cpp
            primitive/hygon/dense_attn_score_thd/bf16_dense_attn_score_thd.cpp
            primitive/hygon/dense_attn_score_thd/fp32_dense_attn_score_thd.cpp
            primitive/hygon/dense_attn_score_thd/mixed_fp32_dense_attn_score_thd.cpp
            primitive/hygon/dense_attn_score_thd/fp16_dense_attn_score_thd.cpp
            primitive/hygon/sparse_attn_target/bf16_sparse_attn_target.cpp
            primitive/hygon/sparse_attn_target/fp32_sparse_attn_target.cpp
            primitive/hygon/sparse_attn_target/mixed_fp32_sparse_attn_target.cpp
            primitive/hygon/sparse_attn_target/fp16_sparse_attn_target.cpp
            primitive/hygon/indexer_topk_thd/fp32_indexer_topk_thd.cpp
            primitive/hygon/indexer_bwd_sparse/bf16_indexer_bwd_sparse.cpp
            primitive/hygon/indexer_bwd_sparse/fp32_indexer_bwd_sparse.cpp
            primitive/hygon/indexer_bwd_sparse/mixed_fp32_indexer_bwd_sparse.cpp
            primitive/hygon/indexer_bwd_sparse/fp16_indexer_bwd_sparse.cpp
            primitive/hygon/indexer_bwd_dense/bf16_indexer_bwd_dense.cpp
            primitive/hygon/indexer_bwd_dense/fp32_indexer_bwd_dense.cpp
            primitive/hygon/indexer_bwd_dense/mixed_fp32_indexer_bwd_dense.cpp
            primitive/hygon/indexer_bwd_dense/fp16_indexer_bwd_dense.cpp
            primitive/hygon/indexer_bwd_dense_thd/bf16_indexer_bwd_dense_thd.cpp
            primitive/hygon/indexer_bwd_dense_thd/fp32_indexer_bwd_dense_thd.cpp
            primitive/hygon/indexer_bwd_dense_thd/mixed_fp32_indexer_bwd_dense_thd.cpp
            primitive/hygon/indexer_bwd_dense_thd/fp16_indexer_bwd_dense_thd.cpp
            primitive/hygon/csa_compressor_fwd/bf16_csa_compressor_fwd.cpp
            primitive/hygon/csa_compressor_bwd/bf16_csa_compressor_bwd.cpp
            primitive/hygon/sinkhorn_fwd/bf16_sinkhorn_fwd.cpp
            primitive/hygon/sinkhorn_fwd/fp32_sinkhorn_fwd.cpp
            primitive/hygon/sinkhorn_fwd/fp16_sinkhorn_fwd.cpp
            primitive/hygon/sinkhorn_bwd/bf16_sinkhorn_bwd.cpp
            primitive/hygon/sinkhorn_bwd/fp32_sinkhorn_bwd.cpp
            primitive/hygon/sinkhorn_bwd/fp16_sinkhorn_bwd.cpp
            primitive/hygon/h_aggregate_fwd/mixed_fp32_h_aggregate_fwd.cpp
            primitive/hygon/h_aggregate_fwd/fp32_h_aggregate_fwd.cpp
            primitive/hygon/h_aggregate_bwd/mixed_fp32_h_aggregate_bwd.cpp
            primitive/hygon/h_aggregate_bwd/fp32_h_aggregate_bwd.cpp
            primitive/hygon/h_post_bda_fwd/bf16_h_post_bda_fwd.cpp
            primitive/hygon/h_post_bda_fwd/fp32_h_post_bda_fwd.cpp
            primitive/hygon/h_post_bda_fwd/fp16_h_post_bda_fwd.cpp
            primitive/hygon/h_post_bda_bwd/bf16_h_post_bda_bwd.cpp
            primitive/hygon/h_post_bda_bwd/fp32_h_post_bda_bwd.cpp
            primitive/hygon/h_post_bda_bwd/fp16_h_post_bda_bwd.cpp
            primitive/hygon/proj_rms_fwd/bf16_proj_rms_fwd.cpp
            primitive/hygon/proj_rms_fwd/fp32_proj_rms_fwd.cpp
            primitive/hygon/proj_rms_fwd/fp16_proj_rms_fwd.cpp
            primitive/hygon/proj_rms_fwd/mixed_fp32_proj_rms_fwd.cpp
            primitive/hygon/proj_rms_bwd/bf16_proj_rms_bwd.cpp
            primitive/hygon/proj_rms_bwd/fp32_proj_rms_bwd.cpp
            primitive/hygon/proj_rms_bwd/fp16_proj_rms_bwd.cpp
            primitive/hygon/proj_rms_bwd/mixed_fp32_proj_rms_bwd.cpp
            primitive/hygon/proj_rms_compute_h_fwd/bf16_proj_rms_compute_h_fwd.cpp
            primitive/hygon/proj_rms_compute_h_fwd/fp32_proj_rms_compute_h_fwd.cpp
            primitive/hygon/proj_rms_compute_h_fwd/fp16_proj_rms_compute_h_fwd.cpp
            primitive/hygon/proj_rms_compute_h_fwd/mixed_fp32_proj_rms_compute_h_fwd.cpp
            primitive/hygon/proj_rms_compute_h_bwd/bf16_proj_rms_compute_h_bwd.cpp
            primitive/hygon/proj_rms_compute_h_bwd/fp32_proj_rms_compute_h_bwd.cpp
            primitive/hygon/proj_rms_compute_h_bwd/fp16_proj_rms_compute_h_bwd.cpp
            primitive/hygon/proj_rms_compute_h_bwd/mixed_fp32_proj_rms_compute_h_bwd.cpp
            primitive/hygon/dense_attn_score/bf16_dense_attn_score.hip
            primitive/hygon/dense_attn_score/fp32_dense_attn_score.hip
            primitive/hygon/dense_attn_score/mixed_fp32_dense_attn_score.hip
            primitive/hygon/dense_attn_score/fp16_dense_attn_score.hip
            primitive/hygon/sparse_attn_fwd/bf16_sparse_attn_fwd.hip
            primitive/hygon/sparse_attn_fwd/fp32_sparse_attn_fwd.hip
            primitive/hygon/sparse_attn_fwd/mixed_fp32_sparse_attn_fwd.hip
            primitive/hygon/sparse_attn_fwd/fp16_sparse_attn_fwd.hip
            primitive/hygon/sparse_indexer_predict/bf16_sparse_indexer_predict.hip
            primitive/hygon/sparse_indexer_predict/fp32_sparse_indexer_predict.hip
            primitive/hygon/sparse_indexer_predict/mixed_fp32_sparse_indexer_predict.hip
            primitive/hygon/sparse_indexer_predict/fp16_sparse_indexer_predict.hip
            primitive/hygon/sparse_attn_bwd/bf16_sparse_attn_bwd.hip
            primitive/hygon/sparse_attn_bwd/fp32_sparse_attn_bwd.hip
            primitive/hygon/sparse_attn_bwd/mixed_fp32_sparse_attn_bwd.hip
            primitive/hygon/sparse_attn_bwd/fp16_sparse_attn_bwd.hip
            primitive/hygon/indexer_fwd/bf16_indexer_fwd.hip
            primitive/hygon/indexer_fwd/fp32_indexer_fwd.hip
            primitive/hygon/indexer_fwd/mixed_fp32_indexer_fwd.hip
            primitive/hygon/indexer_fwd/fp16_indexer_fwd.hip
            primitive/hygon/indexer_fwd_thd/bf16_indexer_fwd_thd.hip
            primitive/hygon/indexer_fwd_thd/fp32_indexer_fwd_thd.hip
            primitive/hygon/indexer_fwd_thd/mixed_fp32_indexer_fwd_thd.hip
            primitive/hygon/indexer_fwd_thd/fp16_indexer_fwd_thd.hip
            primitive/hygon/dense_indexer_score/bf16_dense_indexer_score.hip
            primitive/hygon/dense_indexer_score/fp32_dense_indexer_score.hip
            primitive/hygon/dense_indexer_score/mixed_fp32_dense_indexer_score.hip
            primitive/hygon/dense_indexer_score/fp16_dense_indexer_score.hip
            primitive/hygon/dense_indexer_score_thd/bf16_dense_indexer_score_thd.hip
            primitive/hygon/dense_indexer_score_thd/fp32_dense_indexer_score_thd.hip
            primitive/hygon/dense_indexer_score_thd/mixed_fp32_dense_indexer_score_thd.hip
            primitive/hygon/dense_indexer_score_thd/fp16_dense_indexer_score_thd.hip
            primitive/hygon/dense_attn_score_thd/bf16_dense_attn_score_thd.hip
            primitive/hygon/dense_attn_score_thd/fp32_dense_attn_score_thd.hip
            primitive/hygon/dense_attn_score_thd/mixed_fp32_dense_attn_score_thd.hip
            primitive/hygon/dense_attn_score_thd/fp16_dense_attn_score_thd.hip
            primitive/hygon/sparse_attn_target/bf16_sparse_attn_target.hip
            primitive/hygon/sparse_attn_target/fp32_sparse_attn_target.hip
            primitive/hygon/sparse_attn_target/mixed_fp32_sparse_attn_target.hip
            primitive/hygon/sparse_attn_target/fp16_sparse_attn_target.hip
            primitive/hygon/indexer_topk/fp32_indexer_topk.hip
            primitive/hygon/indexer_topk_thd/fp32_indexer_topk_thd.hip
            primitive/hygon/compactify/int32_compactify.hip
            primitive/hygon/sparse_attn_fwd_indexer/bf16_sparse_attn_fwd_indexer.hip
            primitive/hygon/sparse_attn_fwd_indexer/fp32_sparse_attn_fwd_indexer.hip
            primitive/hygon/sparse_attn_fwd_indexer/mixed_fp32_sparse_attn_fwd_indexer.hip
            primitive/hygon/sparse_attn_fwd_indexer/fp16_sparse_attn_fwd_indexer.hip
            primitive/hygon/indexer_bwd_sparse/bf16_indexer_bwd_sparse.hip
            primitive/hygon/indexer_bwd_sparse/fp32_indexer_bwd_sparse.hip
            primitive/hygon/indexer_bwd_sparse/mixed_fp32_indexer_bwd_sparse.hip
            primitive/hygon/indexer_bwd_sparse/fp16_indexer_bwd_sparse.hip
            primitive/hygon/indexer_bwd_dense/bf16_indexer_bwd_dense.hip
            primitive/hygon/indexer_bwd_dense/fp32_indexer_bwd_dense.hip
            primitive/hygon/indexer_bwd_dense/mixed_fp32_indexer_bwd_dense.hip
            primitive/hygon/indexer_bwd_dense/fp16_indexer_bwd_dense.hip
            primitive/hygon/indexer_bwd_dense_thd/bf16_indexer_bwd_dense_thd.hip
            primitive/hygon/indexer_bwd_dense_thd/fp32_indexer_bwd_dense_thd.hip
            primitive/hygon/indexer_bwd_dense_thd/mixed_fp32_indexer_bwd_dense_thd.hip
            primitive/hygon/indexer_bwd_dense_thd/fp16_indexer_bwd_dense_thd.hip
            primitive/hygon/csa_compressor_fwd/bf16_csa_compressor_fwd.hip
            primitive/hygon/csa_compressor_bwd/bf16_csa_compressor_bwd.hip
            primitive/hygon/sinkhorn_fwd/bf16_sinkhorn_fwd.hip
            primitive/hygon/sinkhorn_fwd/fp32_sinkhorn_fwd.hip
            primitive/hygon/sinkhorn_fwd/fp16_sinkhorn_fwd.hip
            primitive/hygon/sinkhorn_bwd/bf16_sinkhorn_bwd.hip
            primitive/hygon/sinkhorn_bwd/fp32_sinkhorn_bwd.hip
            primitive/hygon/sinkhorn_bwd/fp16_sinkhorn_bwd.hip
            primitive/hygon/h_aggregate_fwd/mixed_fp32_h_aggregate_fwd.hip
            primitive/hygon/h_aggregate_fwd/fp32_h_aggregate_fwd.hip
            primitive/hygon/h_aggregate_bwd/mixed_fp32_h_aggregate_bwd.hip
            primitive/hygon/h_aggregate_bwd/fp32_h_aggregate_bwd.hip
            primitive/hygon/h_post_bda_fwd/bf16_h_post_bda_fwd.hip
            primitive/hygon/h_post_bda_fwd/fp32_h_post_bda_fwd.hip
            primitive/hygon/h_post_bda_fwd/fp16_h_post_bda_fwd.hip
            primitive/hygon/h_post_bda_bwd/bf16_h_post_bda_bwd.hip
            primitive/hygon/h_post_bda_bwd/fp32_h_post_bda_bwd.hip
            primitive/hygon/h_post_bda_bwd/fp16_h_post_bda_bwd.hip
            primitive/hygon/proj_rms_fwd/bf16_proj_rms_fwd.hip
            primitive/hygon/proj_rms_fwd/fp32_proj_rms_fwd.hip
            primitive/hygon/proj_rms_fwd/fp16_proj_rms_fwd.hip
            primitive/hygon/proj_rms_fwd/mixed_fp32_proj_rms_fwd.hip
            primitive/hygon/proj_rms_bwd/bf16_proj_rms_bwd.hip
            primitive/hygon/proj_rms_bwd/fp32_proj_rms_bwd.hip
            primitive/hygon/proj_rms_bwd/fp16_proj_rms_bwd.hip
            primitive/hygon/proj_rms_bwd/mixed_fp32_proj_rms_bwd.hip
            primitive/hygon/proj_rms_compute_h_fwd/bf16_proj_rms_compute_h_fwd.hip
            primitive/hygon/proj_rms_compute_h_fwd/fp32_proj_rms_compute_h_fwd.hip
            primitive/hygon/proj_rms_compute_h_fwd/fp16_proj_rms_compute_h_fwd.hip
            primitive/hygon/proj_rms_compute_h_fwd/mixed_fp32_proj_rms_compute_h_fwd.hip
            primitive/hygon/proj_rms_compute_h_bwd/bf16_proj_rms_compute_h_bwd.hip
            primitive/hygon/proj_rms_compute_h_bwd/fp32_proj_rms_compute_h_bwd.hip
            primitive/hygon/proj_rms_compute_h_bwd/fp16_proj_rms_compute_h_bwd.hip
            primitive/hygon/proj_rms_compute_h_bwd/mixed_fp32_proj_rms_compute_h_bwd.hip
            primitive/hygon/gemm/fp32_gemm.hip)
  # Per-architecture generic availability macros: for every architecture
  # whose complete device target set was built, FTRAIN_HAS_<ARCH_UPPER> is
  # defined (FTRAIN_HAS_GFX936, FTRAIN_HAS_GFX938, ...).  The macros gate
  # both the registered records (family.cpp, compiled here) and the public
  # Finder header, so they are PUBLIC: test and other consumers of
  # finder.hpp must see the same values.  Operator code never invents its
  # own arch macros; it gates arch-specific host code on these generic ones,
  # and the family registration below keeps the registered records in sync
  # with the linked device code objects.  With the default list both macros
  # are defined and selection is purely run time (Constraints::getArch());
  # a single-architecture subset list keeps only that architecture's
  # records and the build stays linkable.
  if(TARGET ftrain_gdn_gfx936 AND TARGET ftrain_gdn_four_gfx936 AND
     TARGET ftrain_gdn_bwd_gfx936 AND TARGET ftrain_gdn2_gfx936 AND
     TARGET ftrain_gdn2_bwd_gfx936)
    target_sources(
      flash_train_impl PRIVATE
      primitive/hygon/gdn/fused_forward.cpp
      primitive/hygon/gdn/four_forward.cpp
      primitive/hygon/gdn/chunk_backward.cpp
      primitive/hygon/gdn2/chunk_forward.cpp
      primitive/hygon/gdn2/chunk_backward.cpp)
    target_compile_definitions(flash_train_impl PUBLIC FTRAIN_HAS_GFX936=1)
  endif()
  if(TARGET ftrain_gdn_gfx938 AND TARGET ftrain_gdn_bwd_gfx938 AND
     TARGET ftrain_gdn2_gfx938 AND TARGET ftrain_gdn2_bwd_gfx938)
    target_sources(
      flash_train_impl PRIVATE
      primitive/hygon/gdn/fused_forward_gfx938.cpp
      primitive/hygon/gdn/chunk_backward_gfx938.cpp
      primitive/hygon/gdn2/chunk_forward_gfx938.cpp
      primitive/hygon/gdn2/chunk_backward_gfx938.cpp)
    target_compile_definitions(flash_train_impl PUBLIC FTRAIN_HAS_GFX938=1)
  endif()
  target_sources(
    flash_train_impl
    PRIVATE primitive/hygon/kda/g2_kda.cpp
            primitive/hygon/kda/g2_runner.cpp
            primitive/hygon/kda/g2_kernels.hip)
  # The G2 KDA kernels use the HIP warp-sync shuffle builtins.
  set_source_files_properties(primitive/hygon/kda/g2_kernels.hip PROPERTIES COMPILE_DEFINITIONS
                                                                       HIP_ENABLE_WARP_SYNC_BUILTINS)
  # GLM D=576/H=64/TopK=2048 backward spills without this option (gfx936).
  # Keep the register-allocation option local to the BF16 backward source.
  set_source_files_properties(
    primitive/hygon/sparse_attn_bwd/bf16_sparse_attn_bwd.hip
    PROPERTIES COMPILE_OPTIONS "-mllvm;-enable-num-vgprs-512=true")
  target_compile_options(flash_train_impl PRIVATE -Wno-return-type-void-func)
  target_link_libraries(flash_train_impl PUBLIC hip::host)
  target_link_libraries(flash_train_impl PRIVATE ROCTX::roctx)
endif()

if(FTRAIN_BUILD_SHARED_LIBS)
  target_compile_definitions(flash_train_impl PRIVATE FTRAIN_BUILDING)
else()
  target_compile_definitions(flash_train_impl PRIVATE FTRAIN_STATIC)
endif()

add_library(flash_train ${FTRAIN_LIBRARY_TYPE})
target_sources(flash_train PRIVATE $<TARGET_OBJECTS:flash_train_impl>)
if(FTRAIN_PLATFORM_HYGON_HIP)
  get_property(ftrain_hygon_operator_targets GLOBAL PROPERTY FTRAIN_HYGON_OPERATOR_TARGETS)
  foreach(ftrain_target IN LISTS ftrain_hygon_operator_targets)
    target_sources(flash_train PRIVATE $<TARGET_OBJECTS:${ftrain_target}>)
  endforeach()
endif()
add_library(flash_train::flash_train ALIAS flash_train)
target_link_libraries(flash_train PRIVATE Threads::Threads)

target_sources(
  flash_train
  PUBLIC FILE_SET
  HEADERS
  BASE_DIRS
  "${PROJECT_BINARY_DIR}/include"
  "${PROJECT_SOURCE_DIR}/include"
  FILES
  "${PROJECT_BINARY_DIR}/include/flash_train/control.h"
  "${PROJECT_BINARY_DIR}/include/flash_train/version.h"
  "${PROJECT_SOURCE_DIR}/include/flash_train/common.h"
  "${PROJECT_SOURCE_DIR}/include/flash_train/env.h"
  "${PROJECT_SOURCE_DIR}/include/flash_train/flash_train.h")

if(FTRAIN_BUILD_SHARED_LIBS)
  # A binding build ships the library inside the wheel as one unversioned
  # file next to the extension module.
  if(NOT FTRAIN_BUILD_WHEEL)
    set_target_properties(
      flash_train PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION
                                                      ${PROJECT_VERSION_MAJOR})
  endif()
else()
  target_compile_definitions(flash_train PUBLIC FTRAIN_STATIC)
endif()

if(FTRAIN_PLATFORM_HYGON_HIP)
  target_link_libraries(flash_train PUBLIC hip::host)
  target_link_libraries(flash_train PRIVATE ROCTX::roctx)
endif()
