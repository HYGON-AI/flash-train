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

/** GDN forward. State views are [B,HV,K,V] fp32 when present, or rank-four
 * zero-element views when absent: any single dimension being zero is
 * sufficient to signal absence (the total element count is what is checked),
 * so the other three dimensions of an absent view carry no meaning and are
 * never read or validated -- callers may put any values there. No primitive
 * is registered in this skeleton; structurally valid arguments return
 * FTRAIN_STATUS_UNSUPPORTED. */
FTRAIN_API FTrainStatus ftrainGdnForward(FTrainStorageView q, FTrainStorageView k, FTrainStorageView v,
                                        FTrainStorageView g, FTrainStorageView beta, FTrainStorageView h0,
                                        FTrainStorageView out, FTrainStorageView ht, float scale, void* workspace,
                                        uint64_t workspace_bytes, FTrainStream stream);
/** GDN backward. h0, dht and dh0 follow the same external state layout. */
FTRAIN_API FTrainStatus ftrainGdnBackward(FTrainStorageView q, FTrainStorageView k, FTrainStorageView v,
                                         FTrainStorageView g, FTrainStorageView beta, FTrainStorageView do_,
                                         FTrainStorageView h0, FTrainStorageView dht, FTrainStorageView dq,
                                         FTrainStorageView dk, FTrainStorageView dv, FTrainStorageView dg,
                                         FTrainStorageView dbeta, FTrainStorageView dh0, float scale, void* workspace,
                                         uint64_t workspace_bytes, FTrainStream stream);
/** GDN2 forward, without state ports. The public chunk length is fixed to 64. */
FTRAIN_API FTrainStatus ftrainGdn2Forward(FTrainStorageView q, FTrainStorageView k, FTrainStorageView v,
                                         FTrainStorageView g, FTrainStorageView b, FTrainStorageView w,
                                         FTrainStorageView out, float scale, void* workspace,
                                         uint64_t workspace_bytes, FTrainStream stream);
/** GDN2 backward, without state ports. The public chunk length is fixed to 64. */
FTRAIN_API FTrainStatus ftrainGdn2Backward(FTrainStorageView q, FTrainStorageView k, FTrainStorageView v,
                                          FTrainStorageView g, FTrainStorageView b, FTrainStorageView w,
                                          FTrainStorageView do_, FTrainStorageView dq, FTrainStorageView dk,
                                          FTrainStorageView dv, FTrainStorageView dg, FTrainStorageView db,
                                          FTrainStorageView dw, float scale, void* workspace,
                                          uint64_t workspace_bytes, FTrainStream stream);
/**
 * @brief The Tensor ports of one KDA forward call (see ftrainPatternAddKdaForward).
 *
 * Packed THD layout: q, k, v, o [T, H, 128] bf16; g [T, H, 128] (bf16 raw gate with
 * gate_in_kernel, else fp32 per-token log decay); beta [T, H] fp32 post-activation; a_log [H]
 * and dt_bias [H, 128] fp32 (gate_in_kernel only); cu_seqlens [N + 1] int32; the saved
 * tensors q_rstd, k_rstd [T, H] fp32 (l2norm_qk only) and aqk, akk [T, H, 64] bf16 are
 * outputs the backward consumes. Unused ports take null memory.
 *
 * Optional outputs the backward can consume instead of recomputing (all bit-identical to the
 * recompute; each costs saved memory and gains backward time):
 * - qn, kn (bf16 [T, H, 128], l2norm_qk only, both or neither): the l2-normalized q / k. The
 *   backward then needs neither q nor k (they may take null memory there), so saving qn / kn
 *   instead of q / k costs no extra memory when the caller does not keep q / k anyway.
 * - h (bf16 [ftrainKdaSavedHElements(T, H, N, has_cu_seqlens)], opaque layout): the recurrent
 *   state h, 512 B per token and head; the backward then runs no recurrence of its own.
 *   Recommended whenever a backward follows and H <= 64 (measured faster on gfx936).
 * - state (bf16 [ftrainKdaStateElements(T, H, N, has_cu_seqlens)], opaque layout): h plus the
 *   backward's v_new operands, 1024 B per token and head. Meant for activation-checkpoint
 *   recompute at H > 64 (the forward runs right before the backward, so the buffer is
 *   short-lived). Exclusive with h.
 *
 * initial_state (optional input) and final_state (optional output), fp32 [N, H, K, V] (N = 1
 * without cu_seqlens): sequence n starts from initial_state[n] instead of the zero state, and
 * its state after its last token is written to final_state[n].
 */
typedef struct {
    FTrainStorageView q, k, v, g, beta, a_log, dt_bias, cu_seqlens;
    FTrainStorageView o, q_rstd, k_rstd, aqk, akk;
    FTrainStorageView qn, kn, h;
    FTrainStorageView state;
    FTrainStorageView initial_state, final_state;
} FTrainKdaForwardTensors;

/** @brief The Tensor ports of one KDA backward call (see ftrainPatternAddKdaBackward); qn, kn,
 * h and state are the forward's optional outputs (null memory when the forward did not write
 * them; with qn / kn given, q / k may take null memory). The outputs dq, dk, dg also serve as
 * scratch during the call and must not alias any input. With qn / kn given and dk directly
 * following dq in memory (one [2, T, H, 128] buffer), the backward also stages its fp32 gate
 * scratch there: the workspace shrinks by T * H * 512 bytes. */
typedef struct {
    FTrainStorageView d_o, q, k, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk;
    FTrainStorageView qn, kn;
    FTrainStorageView dq, dk, dv, dg, dbeta, da_log, ddt_bias;
    FTrainStorageView h;
    FTrainStorageView state;
    /* Optional, fp32 [N, H, K, V]: the forward's initial_state (the recompute starts from it; unused
     * with h given), the final state's gradient dht (absent = zero) and the output dh0, the initial
     * state's gradient. */
    FTrainStorageView initial_state, dht, dh0;
} FTrainKdaBackwardTensors;

/** @brief Chunks of the packed layout for T tokens and N sequences: T / 64 for one sequence of
 * T % 64 == 0 rows, else ceil(T / 64) + N. */
static inline int64_t ftrainKdaChunks(int64_t tokens, int64_t seqs, int has_cu_seqlens) {
    return (!has_cu_seqlens && tokens % 64 == 0) ? tokens / 64 : (tokens + 63) / 64 + (has_cu_seqlens ? seqs : 1);
}

/** @brief Element count of the optional KDA state port: H * ftrainKdaChunks(T, N) * 32768. */
static inline uint64_t ftrainKdaStateElements(int64_t tokens, int64_t heads, int64_t seqs, int has_cu_seqlens) {
    return (uint64_t)heads * (uint64_t)ftrainKdaChunks(tokens, seqs, has_cu_seqlens) * 32768u;
}

/** @brief Element count of the optional KDA h port: H * ftrainKdaChunks(T, N) * 16384. */
static inline uint64_t ftrainKdaSavedHElements(int64_t tokens, int64_t heads, int64_t seqs, int has_cu_seqlens) {
    return (uint64_t)heads * (uint64_t)ftrainKdaChunks(tokens, seqs, has_cu_seqlens) * 16384u;
}

/**
 * @brief Returns the workspace bytes the recommended KDA forward Primitive needs.
 *
 * @return FTRAIN_STATUS_INVALID_ARGUMENT for inconsistent ports or attributes,
 * FTRAIN_STATUS_UNSUPPORTED when no implementation applies, otherwise success.
 */
FTRAIN_API FTrainStatus ftrainKdaForwardGetWorkspaceBytes(const FTrainKdaForwardTensors* tensors,
                                                          FTrainKdaAttributes attributes, uint64_t* workspace_bytes);

/**
 * @brief Executes one KDA forward with the recommended Primitive (asynchronous on stream; the
 * memory and workspace must stay valid until the enqueued work completes).
 */
FTRAIN_API FTrainStatus ftrainKdaForward(const FTrainKdaForwardTensors* tensors, FTrainKdaAttributes attributes,
                                         void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** @brief Returns the workspace bytes the recommended KDA backward Primitive needs. */
FTRAIN_API FTrainStatus ftrainKdaBackwardGetWorkspaceBytes(const FTrainKdaBackwardTensors* tensors,
                                                           FTrainKdaAttributes attributes, uint64_t* workspace_bytes);

/** @brief Executes one KDA backward with the recommended Primitive (asynchronous on stream). */
FTRAIN_API FTrainStatus ftrainKdaBackward(const FTrainKdaBackwardTensors* tensors, FTrainKdaAttributes attributes,
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
