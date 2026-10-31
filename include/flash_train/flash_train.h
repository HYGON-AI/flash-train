// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FLASH_TRAIN_H_
#define FTRAIN_FLASH_TRAIN_H_

#include <flash_train/common.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Executes one GEMM with the recommended Primitive.
 *
 * Convenience wrapper over the staged API: it binds the six Tensor ports
 * and the Gemm attributes internally, creates the Plan through the
 * library's selection cache, and executes primitive 0 -- the recommended
 * default -- on stream. It is equivalent to building one Gemm Pattern,
 * creating Ops, filling Args, creating a Plan, and executing its first
 * Primitive.
 *
 * All staged-API rules apply unchanged: a, b, c, and d are rank-two device
 * Tensors with consistent shapes (c's memory may be null, which drops the
 * beta * c term); alpha and beta are host scalar Tensors; execution is
 * asynchronous on stream, and the referenced memory and workspace must
 * stay valid until the enqueued work completes. The workspace must be at
 * least the recommended Primitive's requirement -- null and zero are
 * accepted when it requires none.
 *
 * @param a Device m-by-k input Tensor.
 * @param b Device k-by-n input Tensor.
 * @param c Device m-by-n input Tensor; its memory may be null.
 * @param d Device m-by-n output Tensor.
 * @param alpha Host scalar Tensor scaling a @ b.
 * @param beta Host scalar Tensor scaling c.
 * @param compute_type Numeric type used for computation.
 * @param workspace Workspace address.
 * @param workspace_bytes Size of workspace in bytes.
 * @param stream Stream used for execution.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when a parameter is inconsistent
 * or the workspace is insufficient; FTRAIN_STATUS_UNSUPPORTED when no
 * implementation satisfies the parameters; otherwise, the status of the
 * operation.
 */
FTRAIN_API FTrainStatus ftrainGemm(FTrainStorageView a, FTrainStorageView b, FTrainStorageView c, FTrainStorageView d,
                                   FTrainStorageView alpha, FTrainStorageView beta, FTrainNumericType compute_type,
                                   void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute IndexerFwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddIndexerFwd/ftrainArgsSetIndexerFwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainIndexerFwd(FTrainStorageView q, FTrainStorageView k, FTrainStorageView w,
                                         FTrainStorageView scores, FTrainStorageView ratio, FTrainStorageView sm_scale,
                                         FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                         FTrainStream stream);

/** Execute IndexerFwdThd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddIndexerFwdThd/ftrainArgsSetIndexerFwdThd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainIndexerFwdThd(FTrainStorageView q, FTrainStorageView k, FTrainStorageView w,
                                            FTrainStorageView cu_seqlens_q, FTrainStorageView cu_seqlens_kv,
                                            FTrainStorageView q_causal_offsets, FTrainStorageView scores,
                                            FTrainStorageView ratio, FTrainStorageView sm_scale,
                                            FTrainStorageView max_seqlen_kv, FTrainNumericType compute_type,
                                            void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute DenseIndexerScore through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddDenseIndexerScore/ftrainArgsSetDenseIndexerScore.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainDenseIndexerScore(FTrainStorageView q, FTrainStorageView k, FTrainStorageView w,
                                                FTrainStorageView out, FTrainStorageView denom, FTrainStorageView ratio,
                                                FTrainStorageView sm_scale, FTrainNumericType compute_type,
                                                void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute DenseIndexerScoreThd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddDenseIndexerScoreThd/ftrainArgsSetDenseIndexerScoreThd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainDenseIndexerScoreThd(FTrainStorageView q, FTrainStorageView k, FTrainStorageView w,
                                                   FTrainStorageView cu_seqlens_q, FTrainStorageView cu_seqlens_kv,
                                                   FTrainStorageView q_causal_offsets, FTrainStorageView out,
                                                   FTrainStorageView denom, FTrainStorageView ratio,
                                                   FTrainStorageView sm_scale, FTrainStorageView max_seqlen_kv,
                                                   FTrainNumericType compute_type, void* workspace,
                                                   uint64_t workspace_bytes, FTrainStream stream);

/** Execute DenseAttnScore through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddDenseAttnScore/ftrainArgsSetDenseAttnScore.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainDenseAttnScore(FTrainStorageView q, FTrainStorageView k, FTrainStorageView lse,
                                             FTrainStorageView out, FTrainStorageView denom, FTrainStorageView ratio,
                                             FTrainStorageView softmax_scale, FTrainNumericType compute_type,
                                             void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute DenseAttnScoreThd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddDenseAttnScoreThd/ftrainArgsSetDenseAttnScoreThd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainDenseAttnScoreThd(FTrainStorageView q, FTrainStorageView k, FTrainStorageView lse,
                                                FTrainStorageView cu_seqlens_q, FTrainStorageView cu_seqlens_kv,
                                                FTrainStorageView q_causal_offsets, FTrainStorageView out,
                                                FTrainStorageView denom, FTrainStorageView ratio,
                                                FTrainStorageView softmax_scale, FTrainStorageView max_seqlen_kv,
                                                FTrainNumericType compute_type, void* workspace,
                                                uint64_t workspace_bytes, FTrainStream stream);

/** Execute SparseIndexerPredict through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddSparseIndexerPredict/ftrainArgsSetSparseIndexerPredict.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainSparseIndexerPredict(FTrainStorageView q, FTrainStorageView k, FTrainStorageView w,
                                                   FTrainStorageView topk_indices, FTrainStorageView predict,
                                                   FTrainStorageView sm_scale, FTrainStorageView topk_indices_global,
                                                   FTrainNumericType compute_type, void* workspace,
                                                   uint64_t workspace_bytes, FTrainStream stream);

/** Execute SparseAttnTarget through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddSparseAttnTarget/ftrainArgsSetSparseAttnTarget.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnTarget(FTrainStorageView q, FTrainStorageView k, FTrainStorageView lse,
                                               FTrainStorageView topk_indices, FTrainStorageView target,
                                               FTrainStorageView softmax_scale, FTrainStorageView topk_indices_global,
                                               FTrainNumericType compute_type, void* workspace,
                                               uint64_t workspace_bytes, FTrainStream stream);

/** Execute SparseAttnFwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddSparseAttnFwd/ftrainArgsSetSparseAttnFwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnFwd(FTrainStorageView q, FTrainStorageView kv, FTrainStorageView attn_sink,
                                            FTrainStorageView topk_indices, FTrainStorageView topk_length,
                                            FTrainStorageView out, FTrainStorageView lse,
                                            FTrainStorageView softmax_scale, FTrainNumericType compute_type,
                                            void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute SparseAttnFwdIndexer through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddSparseAttnFwdIndexer/ftrainArgsSetSparseAttnFwdIndexer.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnFwdIndexer(FTrainStorageView q, FTrainStorageView kv,
                                                   FTrainStorageView attn_sink, FTrainStorageView topk_indices,
                                                   FTrainStorageView topk_length, FTrainStorageView out,
                                                   FTrainStorageView max_logits, FTrainStorageView lse,
                                                   FTrainStorageView lse_indexer, FTrainStorageView softmax_scale,
                                                   FTrainStorageView indexer_topk, FTrainNumericType compute_type,
                                                   void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute SparseAttnBwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddSparseAttnBwd/ftrainArgsSetSparseAttnBwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainSparseAttnBwd(FTrainStorageView q, FTrainStorageView kv, FTrainStorageView out,
                                            FTrainStorageView dout, FTrainStorageView lse, FTrainStorageView attn_sink,
                                            FTrainStorageView topk_indices, FTrainStorageView topk_length,
                                            FTrainStorageView dq, FTrainStorageView dkv, FTrainStorageView dsink,
                                            FTrainStorageView softmax_scale, FTrainNumericType compute_type,
                                            void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute IndexerBwdSparse through the staged selection/cache path.
 * BF16 accepts an FP32 grad_loss scalar in host or device memory. Other
 * implementations require host grad_loss; device values are read on the stream.
 * Ports have the same contract as ftrainPatternAddIndexerBwdSparse/ftrainArgsSetIndexerBwdSparse.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainIndexerBwdSparse(FTrainStorageView index_q, FTrainStorageView weights,
                                               FTrainStorageView index_k, FTrainStorageView attn_score,
                                               FTrainStorageView index_score, FTrainStorageView topk_indices,
                                               FTrainStorageView grad_attn_score, FTrainStorageView dq,
                                               FTrainStorageView dw, FTrainStorageView dk, FTrainStorageView sm_scale,
                                               FTrainStorageView loss_coeff, FTrainStorageView grad_loss,
                                               FTrainStorageView topk_indices_global, FTrainNumericType compute_type,
                                               void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute IndexerBwdDense through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddIndexerBwdDense/ftrainArgsSetIndexerBwdDense.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainIndexerBwdDense(
    FTrainStorageView index_q, FTrainStorageView weights, FTrainStorageView index_k, FTrainStorageView attn_score,
    FTrainStorageView attn_l1norm, FTrainStorageView index_score, FTrainStorageView index_lse,
    FTrainStorageView grad_index_score, FTrainStorageView dq, FTrainStorageView dw, FTrainStorageView dk,
    FTrainStorageView sm_scale, FTrainStorageView loss_coeff, FTrainStorageView grad_loss, FTrainStorageView ratio,
    FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute IndexerBwdDenseThd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddIndexerBwdDenseThd/ftrainArgsSetIndexerBwdDenseThd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainIndexerBwdDenseThd(
    FTrainStorageView index_q, FTrainStorageView weights, FTrainStorageView index_k, FTrainStorageView attn_score,
    FTrainStorageView attn_l1norm, FTrainStorageView index_score, FTrainStorageView index_lse,
    FTrainStorageView cu_seqlens_q, FTrainStorageView cu_seqlens_k, FTrainStorageView q_causal_offsets,
    FTrainStorageView grad_index_score, FTrainStorageView dq, FTrainStorageView dw, FTrainStorageView dk,
    FTrainStorageView sm_scale, FTrainStorageView loss_coeff, FTrainStorageView grad_loss, FTrainStorageView ratio,
    FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute IndexerTopK through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddIndexerTopK/ftrainArgsSetIndexerTopK.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainIndexerTopK(FTrainStorageView input_values, FTrainStorageView seq_lens,
                                          FTrainStorageView out_indices, FTrainStorageView out_values,
                                          FTrainStorageView top_k, FTrainStorageView next_n,
                                          FTrainStorageView order_values, void* workspace, uint64_t workspace_bytes,
                                          FTrainStream stream);

/** Execute IndexerTopKThd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddIndexerTopKThd/ftrainArgsSetIndexerTopKThd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainIndexerTopKThd(FTrainStorageView input_values, FTrainStorageView cu_seqlens_q,
                                             FTrainStorageView cu_seqlens_kv, FTrainStorageView q_causal_offsets,
                                             FTrainStorageView out_indices, FTrainStorageView out_values,
                                             FTrainStorageView ratio, FTrainStorageView top_k,
                                             FTrainStorageView order_values, void* workspace, uint64_t workspace_bytes,
                                             FTrainStream stream);

/** Execute Compactify through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddCompactify/ftrainArgsSetCompactify.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainCompactify(FTrainStorageView indices, FTrainStorageView output,
                                         FTrainStorageView topk_length, void* workspace, uint64_t workspace_bytes,
                                         FTrainStream stream);

/** Execute CSACompressorFwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddCSACompressorFwd/ftrainArgsSetCSACompressorFwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainCSACompressorFwd(FTrainStorageView kv, FTrainStorageView score, FTrainStorageView ape,
                                               FTrainStorageView cu_seqlens, FTrainStorageView cu_seqlens_comp,
                                               FTrainStorageView compressed, FTrainStorageView ratio,
                                               FTrainStorageView head_dim, FTrainStorageView coff,
                                               FTrainStorageView total_comp, FTrainNumericType compute_type,
                                               void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute CSACompressorBwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddCSACompressorBwd/ftrainArgsSetCSACompressorBwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainCSACompressorBwd(FTrainStorageView kv, FTrainStorageView score, FTrainStorageView ape,
                                               FTrainStorageView cu_seqlens, FTrainStorageView cu_seqlens_comp,
                                               FTrainStorageView grad_compressed, FTrainStorageView grad_kv,
                                               FTrainStorageView grad_score, FTrainStorageView grad_ape,
                                               FTrainStorageView ratio, FTrainStorageView head_dim,
                                               FTrainStorageView coff, FTrainNumericType compute_type, void* workspace,
                                               uint64_t workspace_bytes, FTrainStream stream);

/** Execute SinkhornFwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddSinkhornFwd/ftrainArgsSetSinkhornFwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainSinkhornFwd(FTrainStorageView logits, FTrainStorageView output, FTrainStorageView m_init,
                                          FTrainStorageView num_iterations, FTrainStorageView eps,
                                          FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                          FTrainStream stream);

/** Execute SinkhornBwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddSinkhornBwd/ftrainArgsSetSinkhornBwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainSinkhornBwd(FTrainStorageView grad_output, FTrainStorageView m_init,
                                          FTrainStorageView grad_logits, FTrainStorageView num_iterations,
                                          FTrainStorageView eps, FTrainNumericType compute_type, void* workspace,
                                          uint64_t workspace_bytes, FTrainStream stream);

/** Execute HAggregateFwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddHAggregateFwd/ftrainArgsSetHAggregateFwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainHAggregateFwd(FTrainStorageView x, FTrainStorageView h_pre, FTrainStorageView aggregated,
                                            FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                            FTrainStream stream);

/** Execute HAggregateBwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddHAggregateBwd/ftrainArgsSetHAggregateBwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainHAggregateBwd(FTrainStorageView grad_aggregated, FTrainStorageView x,
                                            FTrainStorageView h_pre, FTrainStorageView grad_x,
                                            FTrainStorageView grad_h_pre, FTrainNumericType compute_type,
                                            void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute HPostBdaFwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddHPostBdaFwd/ftrainArgsSetHPostBdaFwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainHPostBdaFwd(FTrainStorageView h_res, FTrainStorageView residual, FTrainStorageView h_post,
                                          FTrainStorageView x, FTrainStorageView bias, FTrainStorageView output,
                                          FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                          FTrainStream stream);

/** Execute HPostBdaBwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddHPostBdaBwd/ftrainArgsSetHPostBdaBwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainHPostBdaBwd(FTrainStorageView grad_output, FTrainStorageView residual,
                                          FTrainStorageView x, FTrainStorageView bias, FTrainStorageView h_res,
                                          FTrainStorageView h_post, FTrainStorageView grad_residual,
                                          FTrainStorageView grad_x, FTrainStorageView grad_h_res,
                                          FTrainStorageView grad_h_post, FTrainStorageView grad_bias,
                                          FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                          FTrainStream stream);

/** Execute ProjRmsFwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddProjRmsFwd/ftrainArgsSetProjRmsFwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainProjRmsFwd(FTrainStorageView x, FTrainStorageView weight, FTrainStorageView proj,
                                         FTrainStorageView r, FTrainStorageView eps, FTrainNumericType compute_type,
                                         void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute ProjRmsBwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddProjRmsBwd/ftrainArgsSetProjRmsBwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainProjRmsBwd(FTrainStorageView grad_proj, FTrainStorageView grad_r, FTrainStorageView x,
                                         FTrainStorageView weight, FTrainStorageView r, FTrainStorageView grad_x,
                                         FTrainStorageView grad_weight, FTrainStorageView eps,
                                         FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                         FTrainStream stream);

/** Execute ProjRmsComputeHFwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddProjRmsComputeHFwd/ftrainArgsSetProjRmsComputeHFwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainProjRmsComputeHFwd(FTrainStorageView x, FTrainStorageView weight,
                                                 FTrainStorageView alpha_pre, FTrainStorageView alpha_post,
                                                 FTrainStorageView alpha_res, FTrainStorageView bias,
                                                 FTrainStorageView h_pre, FTrainStorageView h_post,
                                                 FTrainStorageView h_res, FTrainStorageView r, FTrainStorageView proj,
                                                 FTrainStorageView n, FTrainStorageView eps,
                                                 FTrainStorageView compute_h_eps, FTrainNumericType compute_type,
                                                 void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** Execute ProjRmsComputeHBwd through the staged selection/cache path.
 * Ports have the same contract as ftrainPatternAddProjRmsComputeHBwd/ftrainArgsSetProjRmsComputeHBwd.
 * Workspace and tensor memory must remain valid until stream completion.
 * Returns INVALID_ARGUMENT for malformed ports or insufficient workspace,
 * UNSUPPORTED if no implementation applies, otherwise the execution status.
 */
FTRAIN_API FTrainStatus ftrainProjRmsComputeHBwd(
    FTrainStorageView grad_h_pre, FTrainStorageView grad_h_post, FTrainStorageView grad_h_res, FTrainStorageView grad_r,
    FTrainStorageView h_pre, FTrainStorageView h_post, FTrainStorageView proj, FTrainStorageView r_ch,
    FTrainStorageView x, FTrainStorageView weight, FTrainStorageView alpha_pre, FTrainStorageView alpha_post,
    FTrainStorageView alpha_res, FTrainStorageView grad_x, FTrainStorageView grad_weight, FTrainStorageView grad_alpha,
    FTrainStorageView grad_bias, FTrainStorageView n, FTrainStorageView eps, FTrainStorageView compute_h_eps,
    FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes, FTrainStream stream);

#ifdef __cplusplus
}
#endif

#endif
