// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace flash_train_torch {

// Defined in flash_train_torch_capi.cpp. Plain pointer and integer types keep
// Torch's CUDA-compat headers separate from the HIP runtime headers.
std::uint8_t numericTypeCodeFp32();
std::uint8_t numericTypeCodeFp16();
std::uint8_t numericTypeCodeBf16();
std::uint8_t numericTypeCodeInt32();

unsigned char callBf16SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                           const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                           const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                           const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                           std::uint8_t topk_indices_rank, const void* predict_data,
                                           const std::int64_t* predict_dims, std::uint8_t predict_rank, double sm_scale,
                                           bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                           void* stream);

unsigned char callFp16SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                           const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                           const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                           const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                           std::uint8_t topk_indices_rank, const void* predict_data,
                                           const std::int64_t* predict_dims, std::uint8_t predict_rank, double sm_scale,
                                           bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                           void* stream);

unsigned char callFp32SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                           const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                           const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                           const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                           std::uint8_t topk_indices_rank, const void* predict_data,
                                           const std::int64_t* predict_dims, std::uint8_t predict_rank, double sm_scale,
                                           bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                           void* stream);

unsigned char callMixedFp32SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                                std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                                std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                                const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                                const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                                std::uint8_t topk_indices_rank, const void* predict_data,
                                                const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                                double sm_scale, bool topk_indices_global, void* workspace,
                                                std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                            const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                            const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                            const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                            std::uint8_t topk_indices_rank, const void* predict_data,
                                            const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                            double sm_scale, bool topk_indices_global, std::uint64_t* workspace_bytes);

unsigned char queryFp16SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                            const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                            const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                            const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                            std::uint8_t topk_indices_rank, const void* predict_data,
                                            const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                            double sm_scale, bool topk_indices_global, std::uint64_t* workspace_bytes);

unsigned char queryFp32SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                            const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                            const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                            const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                            std::uint8_t topk_indices_rank, const void* predict_data,
                                            const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                            double sm_scale, bool topk_indices_global, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                                 std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                                 std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                                 const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                                 const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                                 std::uint8_t topk_indices_rank, const void* predict_data,
                                                 const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                                 double sm_scale, bool topk_indices_global,
                                                 std::uint64_t* workspace_bytes);

unsigned char callBf16IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                 const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                 const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                 const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                 int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                 void* stream);

unsigned char callFp16IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                 const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                 const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                 const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                 int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                 void* stream);

unsigned char callFp32IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                 const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                 const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                 const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                 int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                 void* stream);

unsigned char callMixedFp32IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                      std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                      std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                      const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                      const void* scores_data, const std::int64_t* scores_dims,
                                      std::uint8_t scores_rank, int64_t ratio, double sm_scale, void* workspace,
                                      std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                  const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                  const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                  const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                  int64_t ratio, double sm_scale, std::uint64_t* workspace_bytes);

unsigned char queryFp16IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                  const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                  const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                  const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                  int64_t ratio, double sm_scale, std::uint64_t* workspace_bytes);

unsigned char queryFp32IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                  const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                  const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                  const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                  int64_t ratio, double sm_scale, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                       std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                       std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                       const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                       const void* scores_data, const std::int64_t* scores_dims,
                                       std::uint8_t scores_rank, int64_t ratio, double sm_scale,
                                       std::uint64_t* workspace_bytes);

unsigned char callBf16IndexerFwdThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio, double sm_scale,
    int64_t max_seqlen_kv, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16IndexerFwdThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio, double sm_scale,
    int64_t max_seqlen_kv, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32IndexerFwdThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio, double sm_scale,
    int64_t max_seqlen_kv, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32IndexerFwdThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
    const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* scores_data,
    const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16IndexerFwdThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                     const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                     std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                     const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                     const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                     std::uint8_t q_causal_offsets_rank, const void* scores_data,
                                     const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio,
                                     double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes);

unsigned char queryFp16IndexerFwdThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                     const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                     std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                     const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                     const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                     std::uint8_t q_causal_offsets_rank, const void* scores_data,
                                     const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio,
                                     double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes);

unsigned char queryFp32IndexerFwdThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                     const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                     std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                     const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                     const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                     std::uint8_t q_causal_offsets_rank, const void* scores_data,
                                     const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio,
                                     double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32IndexerFwdThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                          std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                          std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                          const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                          const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                          std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                          const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                          const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                          std::uint8_t q_causal_offsets_rank, const void* scores_data,
                                          const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio,
                                          double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes);

unsigned char callBf16DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                        const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                        const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                        int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                        void* stream);

unsigned char callFp16DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                        const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                        const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                        int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                        void* stream);

unsigned char callFp32DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                        const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                        const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                        int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                        void* stream);

unsigned char callMixedFp32DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                             std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                             std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                             const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                             const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                             const void* denom_data, const std::int64_t* denom_dims,
                                             std::uint8_t denom_rank, int64_t ratio, double sm_scale, void* workspace,
                                             std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                         const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                         const void* denom_data, const std::int64_t* denom_dims,
                                         std::uint8_t denom_rank, int64_t ratio, double sm_scale,
                                         std::uint64_t* workspace_bytes);

unsigned char queryFp16DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                         const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                         const void* denom_data, const std::int64_t* denom_dims,
                                         std::uint8_t denom_rank, int64_t ratio, double sm_scale,
                                         std::uint64_t* workspace_bytes);

unsigned char queryFp32DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                         const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                         const void* denom_data, const std::int64_t* denom_dims,
                                         std::uint8_t denom_rank, int64_t ratio, double sm_scale,
                                         std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                              std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                              std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                              const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                              const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                              const void* denom_data, const std::int64_t* denom_dims,
                                              std::uint8_t denom_rank, int64_t ratio, double sm_scale,
                                              std::uint64_t* workspace_bytes);

unsigned char callBf16DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
    const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data, const std::int64_t* denom_dims,
    std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv, void* workspace,
    std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    std::uint64_t* workspace_bytes);

unsigned char queryFp16DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    std::uint64_t* workspace_bytes);

unsigned char queryFp32DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
    const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data, const std::int64_t* denom_dims,
    std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes);

unsigned char callBf16SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                    const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                    const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                    std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                    const void* topk_length_data, const std::int64_t* topk_length_dims,
                                    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                    std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                    std::uint8_t lse_rank, double softmax_scale, void* workspace,
                                    std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                    const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                    const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                    std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                    const void* topk_length_data, const std::int64_t* topk_length_dims,
                                    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                    std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                    std::uint8_t lse_rank, double softmax_scale, void* workspace,
                                    std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                    const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                    const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                    std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                    const void* topk_length_data, const std::int64_t* topk_length_dims,
                                    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                    std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                    std::uint8_t lse_rank, double softmax_scale, void* workspace,
                                    std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         std::uint8_t q_type, const void* kv_data, const std::int64_t* kv_dims,
                                         std::uint8_t kv_rank, std::uint8_t kv_type, const void* attn_sink_data,
                                         const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank,
                                         const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                         std::uint8_t topk_indices_rank, const void* topk_length_data,
                                         const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank,
                                         const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                         std::uint8_t out_type, const void* lse_data, const std::int64_t* lse_dims,
                                         std::uint8_t lse_rank, double softmax_scale, void* workspace,
                                         std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                     std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                     std::uint8_t lse_rank, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char queryFp16SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                     std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                     std::uint8_t lse_rank, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char queryFp32SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                     std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                     std::uint8_t lse_rank, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                          std::uint8_t q_type, const void* kv_data, const std::int64_t* kv_dims,
                                          std::uint8_t kv_rank, std::uint8_t kv_type, const void* attn_sink_data,
                                          const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank,
                                          const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                          std::uint8_t topk_indices_rank, const void* topk_length_data,
                                          const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank,
                                          const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                          std::uint8_t out_type, const void* lse_data, const std::int64_t* lse_dims,
                                          std::uint8_t lse_rank, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char callBf16SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, std::uint8_t kv_type, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, std::uint8_t out_type, const void* max_logits_data,
    const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    std::uint64_t* workspace_bytes);

unsigned char queryFp16SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    std::uint64_t* workspace_bytes);

unsigned char queryFp32SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, std::uint8_t kv_type, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, std::uint8_t out_type, const void* max_logits_data,
    const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    std::uint64_t* workspace_bytes);

unsigned char callBf16SparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* out_data, const std::int64_t* out_dims,
    std::uint8_t out_rank, const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
    std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16SparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* out_data, const std::int64_t* out_dims,
    std::uint8_t out_rank, const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
    std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32SparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* out_data, const std::int64_t* out_dims,
    std::uint8_t out_rank, const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
    std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32SparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, std::uint8_t kv_type, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, std::uint8_t out_type, const void* dout_data,
    const std::int64_t* dout_dims, std::uint8_t dout_rank, std::uint8_t dout_type, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank,
    std::uint8_t dq_type, const void* dkv_data, const std::int64_t* dkv_dims, std::uint8_t dkv_rank,
    std::uint8_t dkv_type, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16SparseAttnBwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims,
                                     std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
                                     std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims,
                                     std::uint8_t dsink_rank, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char queryFp16SparseAttnBwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims,
                                     std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
                                     std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims,
                                     std::uint8_t dsink_rank, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char queryFp32SparseAttnBwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims,
                                     std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
                                     std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims,
                                     std::uint8_t dsink_rank, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32SparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, std::uint8_t kv_type, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, std::uint8_t out_type, const void* dout_data,
    const std::int64_t* dout_dims, std::uint8_t dout_rank, std::uint8_t dout_type, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank,
    std::uint8_t dq_type, const void* dkv_data, const std::int64_t* dkv_dims, std::uint8_t dkv_rank,
    std::uint8_t dkv_type, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char callBf16DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                     int64_t ratio, double softmax_scale, void* workspace,
                                     std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                     int64_t ratio, double softmax_scale, void* workspace,
                                     std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                     int64_t ratio, double softmax_scale, void* workspace,
                                     std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                          std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                          std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
                                          const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* out_data,
                                          const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                          const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                          double softmax_scale, void* workspace, std::uint64_t workspace_bytes,
                                          void* stream);

unsigned char queryBf16DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                      const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                      const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                      const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                      const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                      int64_t ratio, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char queryFp16DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                      const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                      const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                      const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                      const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                      int64_t ratio, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char queryFp32DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                      const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                      const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                      const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                      const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                      int64_t ratio, double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                           std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                           std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
                                           const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* out_data,
                                           const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                           const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                           double softmax_scale, std::uint64_t* workspace_bytes);

unsigned char callBf16DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data, const std::int64_t* denom_dims,
    std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv, void* workspace,
    std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16DenseAttnScoreThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                         const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                         std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                         const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                         const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                         std::uint8_t q_causal_offsets_rank, const void* out_data,
                                         const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                         const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                         double softmax_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes);

unsigned char queryFp16DenseAttnScoreThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                         const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                         std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                         const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                         const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                         std::uint8_t q_causal_offsets_rank, const void* out_data,
                                         const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                         const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                         double softmax_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes);

unsigned char queryFp32DenseAttnScoreThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                         const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                         std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                         const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                         const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                         std::uint8_t q_causal_offsets_rank, const void* out_data,
                                         const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                         const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                         double softmax_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data, const std::int64_t* denom_dims,
    std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv,
    std::uint64_t* workspace_bytes);

unsigned char callBf16SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                       const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                       const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                       const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                       std::uint8_t topk_indices_rank, const void* target_data,
                                       const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                       bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                       void* stream);

unsigned char callFp16SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                       const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                       const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                       const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                       std::uint8_t topk_indices_rank, const void* target_data,
                                       const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                       bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                       void* stream);

unsigned char callFp32SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                       const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                       const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                       const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                       std::uint8_t topk_indices_rank, const void* target_data,
                                       const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                       bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                       void* stream);

unsigned char callMixedFp32SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                            std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                            std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
                                            const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                            const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                            std::uint8_t topk_indices_rank, const void* target_data,
                                            const std::int64_t* target_dims, std::uint8_t target_rank,
                                            double softmax_scale, bool topk_indices_global, void* workspace,
                                            std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                        const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                        std::uint8_t topk_indices_rank, const void* target_data,
                                        const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                        bool topk_indices_global, std::uint64_t* workspace_bytes);

unsigned char queryFp16SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                        const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                        std::uint8_t topk_indices_rank, const void* target_data,
                                        const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                        bool topk_indices_global, std::uint64_t* workspace_bytes);

unsigned char queryFp32SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                        const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                        std::uint8_t topk_indices_rank, const void* target_data,
                                        const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                        bool topk_indices_global, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                             std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                             std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
                                             const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                             const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                             std::uint8_t topk_indices_rank, const void* target_data,
                                             const std::int64_t* target_dims, std::uint8_t target_rank,
                                             double softmax_scale, bool topk_indices_global,
                                             std::uint64_t* workspace_bytes);

unsigned char callFp32IndexerTopK(const void* input_values_data, const std::int64_t* input_values_dims,
                                  std::uint8_t input_values_rank, const void* seq_lens_data,
                                  const std::int64_t* seq_lens_dims, std::uint8_t seq_lens_rank,
                                  const void* out_indices_data, const std::int64_t* out_indices_dims,
                                  std::uint8_t out_indices_rank, const void* out_values_data,
                                  const std::int64_t* out_values_dims, std::uint8_t out_values_rank, int64_t top_k,
                                  int64_t next_n, bool order_values, void* stream);

unsigned char callFp32IndexerTopKThd(const void* input_values_data, const std::int64_t* input_values_dims,
                                     std::uint8_t input_values_rank, const void* cu_seqlens_q_data,
                                     const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
                                     const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims,
                                     std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
                                     const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
                                     const void* out_indices_data, const std::int64_t* out_indices_dims,
                                     std::uint8_t out_indices_rank, const void* out_values_data,
                                     const std::int64_t* out_values_dims, std::uint8_t out_values_rank, int64_t ratio,
                                     int64_t top_k, bool order_values, void* stream);

unsigned char callInt32Compactify(const void* indices_data, const std::int64_t* indices_dims, std::uint8_t indices_rank,
                                  const void* output_data, const std::int64_t* output_dims, std::uint8_t output_rank,
                                  const void* topk_length_data, const std::int64_t* topk_length_dims,
                                  std::uint8_t topk_length_rank, void* stream);

unsigned char callBf16IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
    void* stream, const void* grad_loss_data);

unsigned char callFp16IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
    void* stream);

unsigned char callFp32IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
    void* stream);

unsigned char callMixedFp32IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* topk_indices_data, const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
    const void* grad_attn_score_data, const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, std::uint64_t* workspace_bytes,
    const void* grad_loss_data);

unsigned char queryFp16IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, std::uint64_t* workspace_bytes);

unsigned char queryFp32IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* topk_indices_data, const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
    const void* grad_attn_score_data, const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, bool topk_indices_global, std::uint64_t* workspace_bytes);

unsigned char callBf16IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* attn_l1norm_data, const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* index_lse_data, const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank,
    const void* grad_index_score_data, const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes);

unsigned char queryFp16IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes);

unsigned char queryFp32IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* attn_l1norm_data, const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* index_lse_data, const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank,
    const void* grad_index_score_data, const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes);

unsigned char callBf16IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* attn_l1norm_data, const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* index_lse_data, const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_k_data, const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* grad_index_score_data, const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes);

unsigned char queryFp16IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes);

unsigned char queryFp32IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* attn_l1norm_data, const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* index_lse_data, const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_k_data, const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* grad_index_score_data, const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes);

unsigned char callBf16CSACompressorFwd(const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                       const void* score_data, const std::int64_t* score_dims, std::uint8_t score_rank,
                                       const void* ape_data, const std::int64_t* ape_dims, std::uint8_t ape_rank,
                                       const void* cu_seqlens_data, const std::int64_t* cu_seqlens_dims,
                                       std::uint8_t cu_seqlens_rank, const void* cu_seqlens_comp_data,
                                       const std::int64_t* cu_seqlens_comp_dims, std::uint8_t cu_seqlens_comp_rank,
                                       const void* compressed_data, const std::int64_t* compressed_dims,
                                       std::uint8_t compressed_rank, int64_t ratio, int64_t head_dim, int64_t coff,
                                       int64_t total_comp, void* stream);

unsigned char callBf16CSACompressorBwd(
    const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* score_data,
    const std::int64_t* score_dims, std::uint8_t score_rank, const void* ape_data, const std::int64_t* ape_dims,
    std::uint8_t ape_rank, const void* cu_seqlens_data, const std::int64_t* cu_seqlens_dims,
    std::uint8_t cu_seqlens_rank, const void* cu_seqlens_comp_data, const std::int64_t* cu_seqlens_comp_dims,
    std::uint8_t cu_seqlens_comp_rank, const void* grad_compressed_data, const std::int64_t* grad_compressed_dims,
    std::uint8_t grad_compressed_rank, const void* grad_kv_data, const std::int64_t* grad_kv_dims,
    std::uint8_t grad_kv_rank, const void* grad_score_data, const std::int64_t* grad_score_dims,
    std::uint8_t grad_score_rank, const void* grad_ape_data, const std::int64_t* grad_ape_dims,
    std::uint8_t grad_ape_rank, int64_t ratio, int64_t head_dim, int64_t coff, void* stream);

unsigned char callBf16SinkhornFwd(const void* logits_data, const std::int64_t* logits_dims, std::uint8_t logits_rank,
                                  const void* output_data, const std::int64_t* output_dims, std::uint8_t output_rank,
                                  const void* m_init_data, const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  int64_t num_iterations, double eps, void* stream);

unsigned char callFp16SinkhornFwd(const void* logits_data, const std::int64_t* logits_dims, std::uint8_t logits_rank,
                                  const void* output_data, const std::int64_t* output_dims, std::uint8_t output_rank,
                                  const void* m_init_data, const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  int64_t num_iterations, double eps, void* stream);

unsigned char callFp32SinkhornFwd(const void* logits_data, const std::int64_t* logits_dims, std::uint8_t logits_rank,
                                  const void* output_data, const std::int64_t* output_dims, std::uint8_t output_rank,
                                  const void* m_init_data, const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  int64_t num_iterations, double eps, void* stream);

unsigned char callBf16SinkhornBwd(const void* grad_output_data, const std::int64_t* grad_output_dims,
                                  std::uint8_t grad_output_rank, const void* m_init_data,
                                  const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  const void* grad_logits_data, const std::int64_t* grad_logits_dims,
                                  std::uint8_t grad_logits_rank, int64_t num_iterations, double eps, void* stream);

unsigned char callFp16SinkhornBwd(const void* grad_output_data, const std::int64_t* grad_output_dims,
                                  std::uint8_t grad_output_rank, const void* m_init_data,
                                  const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  const void* grad_logits_data, const std::int64_t* grad_logits_dims,
                                  std::uint8_t grad_logits_rank, int64_t num_iterations, double eps, void* stream);

unsigned char callFp32SinkhornBwd(const void* grad_output_data, const std::int64_t* grad_output_dims,
                                  std::uint8_t grad_output_rank, const void* m_init_data,
                                  const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  const void* grad_logits_data, const std::int64_t* grad_logits_dims,
                                  std::uint8_t grad_logits_rank, int64_t num_iterations, double eps, void* stream);

unsigned char callFp32HAggregateFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                    const void* h_pre_data, const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank,
                                    const void* aggregated_data, const std::int64_t* aggregated_dims,
                                    std::uint8_t aggregated_rank, void* stream);

unsigned char callMixedFp32HAggregateFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                         std::uint8_t x_type, const void* h_pre_data, const std::int64_t* h_pre_dims,
                                         std::uint8_t h_pre_rank, const void* aggregated_data,
                                         const std::int64_t* aggregated_dims, std::uint8_t aggregated_rank,
                                         std::uint8_t aggregated_type, void* stream);

unsigned char callFp32HAggregateBwd(const void* grad_aggregated_data, const std::int64_t* grad_aggregated_dims,
                                    std::uint8_t grad_aggregated_rank, const void* x_data, const std::int64_t* x_dims,
                                    std::uint8_t x_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
                                    std::uint8_t h_pre_rank, const void* grad_x_data, const std::int64_t* grad_x_dims,
                                    std::uint8_t grad_x_rank, const void* grad_h_pre_data,
                                    const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank, void* stream);

unsigned char callMixedFp32HAggregateBwd(
    const void* grad_aggregated_data, const std::int64_t* grad_aggregated_dims, std::uint8_t grad_aggregated_rank,
    std::uint8_t grad_aggregated_type, const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
    std::uint8_t x_type, const void* h_pre_data, const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank,
    const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, std::uint8_t grad_x_type,
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank, void* stream);

unsigned char callBf16HPostBdaFwd(const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
                                  const void* residual_data, const std::int64_t* residual_dims,
                                  std::uint8_t residual_rank, const void* h_post_data, const std::int64_t* h_post_dims,
                                  std::uint8_t h_post_rank, const void* x_data, const std::int64_t* x_dims,
                                  std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
                                  std::uint8_t bias_rank, const void* output_data, const std::int64_t* output_dims,
                                  std::uint8_t output_rank, void* stream);

unsigned char callFp16HPostBdaFwd(const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
                                  const void* residual_data, const std::int64_t* residual_dims,
                                  std::uint8_t residual_rank, const void* h_post_data, const std::int64_t* h_post_dims,
                                  std::uint8_t h_post_rank, const void* x_data, const std::int64_t* x_dims,
                                  std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
                                  std::uint8_t bias_rank, const void* output_data, const std::int64_t* output_dims,
                                  std::uint8_t output_rank, void* stream);

unsigned char callFp32HPostBdaFwd(const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
                                  const void* residual_data, const std::int64_t* residual_dims,
                                  std::uint8_t residual_rank, const void* h_post_data, const std::int64_t* h_post_dims,
                                  std::uint8_t h_post_rank, const void* x_data, const std::int64_t* x_dims,
                                  std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
                                  std::uint8_t bias_rank, const void* output_data, const std::int64_t* output_dims,
                                  std::uint8_t output_rank, void* stream);

unsigned char callBf16HPostBdaBwd(
    const void* grad_output_data, const std::int64_t* grad_output_dims, std::uint8_t grad_output_rank,
    const void* residual_data, const std::int64_t* residual_dims, std::uint8_t residual_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
    std::uint8_t bias_rank, const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, const void* grad_residual_data,
    const std::int64_t* grad_residual_dims, std::uint8_t grad_residual_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, const void* grad_h_post_data,
    const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank, const void* grad_bias_data,
    const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, void* stream);

unsigned char callFp16HPostBdaBwd(
    const void* grad_output_data, const std::int64_t* grad_output_dims, std::uint8_t grad_output_rank,
    const void* residual_data, const std::int64_t* residual_dims, std::uint8_t residual_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
    std::uint8_t bias_rank, const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, const void* grad_residual_data,
    const std::int64_t* grad_residual_dims, std::uint8_t grad_residual_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, const void* grad_h_post_data,
    const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank, const void* grad_bias_data,
    const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, void* stream);

unsigned char callFp32HPostBdaBwd(
    const void* grad_output_data, const std::int64_t* grad_output_dims, std::uint8_t grad_output_rank,
    const void* residual_data, const std::int64_t* residual_dims, std::uint8_t residual_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
    std::uint8_t bias_rank, const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, const void* grad_residual_data,
    const std::int64_t* grad_residual_dims, std::uint8_t grad_residual_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, const void* grad_h_post_data,
    const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank, const void* grad_bias_data,
    const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, void* stream);

unsigned char callBf16ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                 const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                 const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                 const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                 void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                 const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                 const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                 const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                 void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                 const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                 const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                 const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                 void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                      std::uint8_t x_type, const void* weight_data, const std::int64_t* weight_dims,
                                      std::uint8_t weight_rank, std::uint8_t weight_type, const void* proj_data,
                                      const std::int64_t* proj_dims, std::uint8_t proj_rank, std::uint8_t proj_type,
                                      const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank,
                                      std::uint8_t r_type, double eps, void* workspace, std::uint64_t workspace_bytes,
                                      void* stream);

unsigned char queryBf16ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                  const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                  const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                  const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                  std::uint64_t* workspace_bytes);

unsigned char queryFp16ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                  const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                  const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                  const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                  std::uint64_t* workspace_bytes);

unsigned char queryFp32ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                  const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                  const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                  const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                  std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                       std::uint8_t x_type, const void* weight_data, const std::int64_t* weight_dims,
                                       std::uint8_t weight_rank, std::uint8_t weight_type, const void* proj_data,
                                       const std::int64_t* proj_dims, std::uint8_t proj_rank, std::uint8_t proj_type,
                                       const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank,
                                       std::uint8_t r_type, double eps, std::uint64_t* workspace_bytes);

unsigned char callBf16ProjRmsBwd(const void* grad_proj_data, const std::int64_t* grad_proj_dims,
                                 std::uint8_t grad_proj_rank, const void* grad_r_data, const std::int64_t* grad_r_dims,
                                 std::uint8_t grad_r_rank, const void* x_data, const std::int64_t* x_dims,
                                 std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
                                 std::uint8_t weight_rank, const void* r_data, const std::int64_t* r_dims,
                                 std::uint8_t r_rank, const void* grad_x_data, const std::int64_t* grad_x_dims,
                                 std::uint8_t grad_x_rank, const void* grad_weight_data,
                                 const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, double eps,
                                 void* stream);

unsigned char callFp16ProjRmsBwd(const void* grad_proj_data, const std::int64_t* grad_proj_dims,
                                 std::uint8_t grad_proj_rank, const void* grad_r_data, const std::int64_t* grad_r_dims,
                                 std::uint8_t grad_r_rank, const void* x_data, const std::int64_t* x_dims,
                                 std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
                                 std::uint8_t weight_rank, const void* r_data, const std::int64_t* r_dims,
                                 std::uint8_t r_rank, const void* grad_x_data, const std::int64_t* grad_x_dims,
                                 std::uint8_t grad_x_rank, const void* grad_weight_data,
                                 const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, double eps,
                                 void* stream);

unsigned char callFp32ProjRmsBwd(const void* grad_proj_data, const std::int64_t* grad_proj_dims,
                                 std::uint8_t grad_proj_rank, const void* grad_r_data, const std::int64_t* grad_r_dims,
                                 std::uint8_t grad_r_rank, const void* x_data, const std::int64_t* x_dims,
                                 std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
                                 std::uint8_t weight_rank, const void* r_data, const std::int64_t* r_dims,
                                 std::uint8_t r_rank, const void* grad_x_data, const std::int64_t* grad_x_dims,
                                 std::uint8_t grad_x_rank, const void* grad_weight_data,
                                 const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, double eps,
                                 void* stream);

unsigned char callMixedFp32ProjRmsBwd(
    const void* grad_proj_data, const std::int64_t* grad_proj_dims, std::uint8_t grad_proj_rank,
    std::uint8_t grad_proj_type, const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank,
    std::uint8_t grad_r_type, const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type,
    const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type,
    const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, std::uint8_t r_type, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, std::uint8_t grad_x_type, const void* grad_weight_data,
    const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, std::uint8_t grad_weight_type, double eps,
    void* stream);

unsigned char callBf16ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, void* workspace, std::uint64_t workspace_bytes,
    void* stream);

unsigned char callFp16ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, void* workspace, std::uint64_t workspace_bytes,
    void* stream);

unsigned char callFp32ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, void* workspace, std::uint64_t workspace_bytes,
    void* stream);

unsigned char callMixedFp32ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, std::uint8_t h_pre_type, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, std::uint8_t h_post_type, const void* h_res_data, const std::int64_t* h_res_dims,
    std::uint8_t h_res_rank, std::uint8_t h_res_type, const void* r_data, const std::int64_t* r_dims,
    std::uint8_t r_rank, std::uint8_t r_type, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, std::uint8_t proj_type, int64_t n, double eps, double compute_h_eps, void* workspace,
    std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, std::uint64_t* workspace_bytes);

unsigned char queryFp16ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, std::uint64_t* workspace_bytes);

unsigned char queryFp32ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, std::uint8_t h_pre_type, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, std::uint8_t h_post_type, const void* h_res_data, const std::int64_t* h_res_dims,
    std::uint8_t h_res_rank, std::uint8_t h_res_type, const void* r_data, const std::int64_t* r_dims,
    std::uint8_t r_rank, std::uint8_t r_type, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, std::uint8_t proj_type, int64_t n, double eps, double compute_h_eps,
    std::uint64_t* workspace_bytes);

unsigned char callBf16ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp16ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callFp32ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char callMixedFp32ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    std::uint8_t grad_h_pre_type, const void* grad_h_post_data, const std::int64_t* grad_h_post_dims,
    std::uint8_t grad_h_post_rank, std::uint8_t grad_h_post_type, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, std::uint8_t grad_h_res_type,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, std::uint8_t grad_r_type,
    const void* h_pre_data, const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, std::uint8_t h_pre_type,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, std::uint8_t h_post_type,
    const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank, std::uint8_t proj_type,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, std::uint8_t r_ch_type,
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, std::uint8_t grad_x_type, const void* grad_weight_data,
    const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, std::uint8_t grad_weight_type,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, void* workspace, std::uint64_t workspace_bytes, void* stream);

unsigned char queryBf16ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, std::uint64_t* workspace_bytes);

unsigned char queryFp16ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, std::uint64_t* workspace_bytes);

unsigned char queryFp32ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, std::uint64_t* workspace_bytes);

unsigned char queryMixedFp32ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    std::uint8_t grad_h_pre_type, const void* grad_h_post_data, const std::int64_t* grad_h_post_dims,
    std::uint8_t grad_h_post_rank, std::uint8_t grad_h_post_type, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, std::uint8_t grad_h_res_type,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, std::uint8_t grad_r_type,
    const void* h_pre_data, const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, std::uint8_t h_pre_type,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, std::uint8_t h_post_type,
    const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank, std::uint8_t proj_type,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, std::uint8_t r_ch_type,
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, std::uint8_t grad_x_type, const void* grad_weight_data,
    const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, std::uint8_t grad_weight_type,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, std::uint64_t* workspace_bytes);

// Defined in flash_train_torch_capi.cpp. The bridge keeps this torch-facing
// translation unit free of the library's HIP-flavored headers: the HCU
// torch build pulls DTK's CUDA-compat headers, which clash with the HIP
// runtime headers inside one translation unit. Plain pointer and integer
// types only.
unsigned char callFp32Gemm(const void* a, const std::int64_t* a_dims, const void* b, const std::int64_t* b_dims,
                           const void* c, const std::int64_t* c_dims, void* d, const std::int64_t* d_dims,
                           const void* alpha, const void* beta, void* stream);

}  // namespace flash_train_torch

namespace {
using namespace flash_train_torch;

// Local Torch helpers for tensor views, dtype mapping, workspace allocation,
// and status reporting.

inline std::uint8_t numericTypeCode(const at::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::kFloat: return numericTypeCodeFp32();
        case at::kHalf: return numericTypeCodeFp16();
        case at::kBFloat16: return numericTypeCodeBf16();
        case at::kInt: return numericTypeCodeInt32();
        default: TORCH_CHECK(false, "flash_train.torch: unsupported tensor dtype ", tensor.scalar_type());
    }
    return numericTypeCodeFp32();
}

inline void checkDeviceTensor(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a device tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// The HCU torch wheel's C++ stream headers cannot be included together
// with this TU's world (its CUDA-compat and HIP runtime headers clash), so
// ask torch itself for the current stream. The bound callable is cached
// and intentionally never destroyed.
void* currentStream() {
    static const py::object* const current_stream =
        new py::object(py::module_::import("torch").attr("cuda").attr("current_stream"));
    return reinterpret_cast<void*>(py::cast<std::uintptr_t>((*current_stream)().attr("cuda_stream")));
}

void checkMatrix(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a device tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be FP32");
    TORCH_CHECK(tensor.dim() == 2, name, " must be rank two");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

at::Tensor gemm(const at::Tensor& a, const at::Tensor& b, const std::optional<at::Tensor>& c, double alpha,
                double beta) {
    checkMatrix(a, "a");
    checkMatrix(b, "b");
    TORCH_CHECK(a.size(1) == b.size(0), "a's reduction depth must equal b's rows");
    if (c.has_value()) {
        checkMatrix(*c, "c");
        TORCH_CHECK(c->size(0) == a.size(0) && c->size(1) == b.size(1), "c must be shaped m-by-n");
    }

    at::Tensor d = torch::empty({a.size(0), b.size(1)}, a.options());

    const std::int64_t a_dims[2]  = {a.size(0), a.size(1)};
    const std::int64_t b_dims[2]  = {b.size(0), b.size(1)};
    const std::int64_t cd_dims[2] = {a.size(0), b.size(1)};

    float alpha_value = static_cast<float>(alpha);
    float beta_value  = static_cast<float>(beta);

    void* stream = currentStream();
    // FTRAIN_STATUS_SUCCESS is 0; any other code is a failure.
    const unsigned char status = flash_train_torch::callFp32Gemm(
        a.data_ptr(), a_dims, b.data_ptr(), b_dims, c.has_value() ? c->data_ptr() : nullptr, cd_dims, d.data_ptr(),
        cd_dims, &alpha_value, &beta_value, stream);
    TORCH_CHECK(status == 0, "ftrainGemm failed with status ", static_cast<unsigned int>(status));
    return d;
}

std::vector<at::Tensor> compactify(const at::Tensor& indices) {
    checkDeviceTensor(indices, "indices");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(indices.device().type()).getDevice() ==
                                                        indices.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(indices.device()));
    TORCH_CHECK(indices.dim() == 2, "compactify expects (n_rows,topk) int32 indices");
    at::Tensor output           = torch::empty_like(indices);
    at::Tensor length           = torch::empty({indices.size(0)}, indices.options().dtype(at::kInt));
    const auto indices_dims     = indices.sizes().vec();
    const auto output_dims      = output.sizes().vec();
    const auto topk_length_dims = length.sizes().vec();
    const unsigned char status  = [&]() -> unsigned char {
        if ((indices).scalar_type() == at::kInt && (output).scalar_type() == at::kInt &&
            (length).scalar_type() == at::kInt) {
            return callInt32Compactify(
                indices.data_ptr(), indices_dims.data(), static_cast<std::uint8_t>(indices_dims.size()),
                output.data_ptr(), output_dims.data(), static_cast<std::uint8_t>(output_dims.size()), length.data_ptr(),
                topk_length_dims.data(), static_cast<std::uint8_t>(topk_length_dims.size()), currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.compactify failed with status ", static_cast<unsigned int>(status));
    return {output, length};
}

std::vector<at::Tensor> csaCompressorBwd(const at::Tensor& kv, const at::Tensor& score, const at::Tensor& ape,
                                         const at::Tensor& cu_seqlens, const at::Tensor& cu_seqlens_comp,
                                         const at::Tensor& grad_compressed, std::int64_t ratio, std::int64_t head_dim,
                                         std::int64_t coff) {
    checkDeviceTensor(kv, "kv");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(kv.device().type()).getDevice() ==
                                                        kv.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(kv.device()));
    checkDeviceTensor(score, "score");
    checkDeviceTensor(ape, "ape");
    checkDeviceTensor(cu_seqlens, "cu_seqlens");
    checkDeviceTensor(cu_seqlens_comp, "cu_seqlens_comp");
    checkDeviceTensor(grad_compressed, "grad_compressed");
    TORCH_CHECK(kv.dim() == 2 && score.dim() == 2 && grad_compressed.dim() == 2,
                "csa_compressor_bwd expects kv/score (total_tokens, coff*head_dim) and grad (total_comp, head_dim)");
    at::Tensor grad_kv              = torch::empty_like(kv);
    at::Tensor grad_score           = torch::empty_like(score);
    at::Tensor grad_ape             = torch::empty_like(ape);
    const auto kv_dims              = kv.sizes().vec();
    const auto score_dims           = score.sizes().vec();
    const auto ape_dims             = ape.sizes().vec();
    const auto cu_seqlens_dims      = cu_seqlens.sizes().vec();
    const auto cu_seqlens_comp_dims = cu_seqlens_comp.sizes().vec();
    const auto grad_compressed_dims = grad_compressed.sizes().vec();
    const auto grad_kv_dims         = grad_kv.sizes().vec();
    const auto grad_score_dims      = grad_score.sizes().vec();
    const auto grad_ape_dims        = grad_ape.sizes().vec();
    const unsigned char status      = [&]() -> unsigned char {
        if ((kv).scalar_type() == at::kBFloat16 && (score).scalar_type() == at::kBFloat16 &&
            (ape).scalar_type() == at::kFloat && (cu_seqlens).scalar_type() == at::kInt &&
            (cu_seqlens_comp).scalar_type() == at::kInt && (grad_compressed).scalar_type() == at::kBFloat16 &&
            (grad_kv).scalar_type() == at::kBFloat16 && (grad_score).scalar_type() == at::kBFloat16 &&
            (grad_ape).scalar_type() == at::kFloat) {
            return callBf16CSACompressorBwd(
                kv.data_ptr(), kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), score.data_ptr(),
                score_dims.data(), static_cast<std::uint8_t>(score_dims.size()), ape.data_ptr(), ape_dims.data(),
                static_cast<std::uint8_t>(ape_dims.size()), cu_seqlens.data_ptr(), cu_seqlens_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_dims.size()), cu_seqlens_comp.data_ptr(),
                cu_seqlens_comp_dims.data(), static_cast<std::uint8_t>(cu_seqlens_comp_dims.size()),
                grad_compressed.data_ptr(), grad_compressed_dims.data(),
                static_cast<std::uint8_t>(grad_compressed_dims.size()), grad_kv.data_ptr(), grad_kv_dims.data(),
                static_cast<std::uint8_t>(grad_kv_dims.size()), grad_score.data_ptr(), grad_score_dims.data(),
                static_cast<std::uint8_t>(grad_score_dims.size()), grad_ape.data_ptr(), grad_ape_dims.data(),
                static_cast<std::uint8_t>(grad_ape_dims.size()), ratio, head_dim, coff, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.csa_compressor_bwd failed with status ",
                static_cast<unsigned int>(status));
    return {grad_kv, grad_score, grad_ape};
}

at::Tensor csaCompressorFwd(const at::Tensor& kv, const at::Tensor& score, const at::Tensor& ape,
                            const at::Tensor& cu_seqlens, const at::Tensor& cu_seqlens_comp, std::int64_t ratio,
                            std::int64_t head_dim, std::int64_t coff, std::int64_t total_comp) {
    checkDeviceTensor(kv, "kv");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(kv.device().type()).getDevice() ==
                                                        kv.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(kv.device()));
    checkDeviceTensor(score, "score");
    checkDeviceTensor(ape, "ape");
    checkDeviceTensor(cu_seqlens, "cu_seqlens");
    checkDeviceTensor(cu_seqlens_comp, "cu_seqlens_comp");
    TORCH_CHECK(kv.dim() == 2 && score.dim() == 2 && ape.dim() == 2,
                "csa_compressor_fwd expects kv/score (total_tokens, coff*head_dim) and ape (ratio, coff*head_dim)");
    at::Tensor compressed           = torch::empty({total_comp, head_dim}, kv.options());
    const auto kv_dims              = kv.sizes().vec();
    const auto score_dims           = score.sizes().vec();
    const auto ape_dims             = ape.sizes().vec();
    const auto cu_seqlens_dims      = cu_seqlens.sizes().vec();
    const auto cu_seqlens_comp_dims = cu_seqlens_comp.sizes().vec();
    const auto compressed_dims =
        (!(total_comp == 0)) ? compressed.sizes().vec() : std::vector<std::int64_t>{0, head_dim};
    const void* compressed_data = (!(total_comp == 0)) ? compressed.data_ptr() : nullptr;
    const auto compressed_type  = (!(total_comp == 0)) ? (compressed).scalar_type() : (kv).scalar_type();
    const unsigned char status  = [&]() -> unsigned char {
        if ((kv).scalar_type() == at::kBFloat16 && (score).scalar_type() == at::kBFloat16 &&
            (ape).scalar_type() == at::kFloat && (cu_seqlens).scalar_type() == at::kInt &&
            (cu_seqlens_comp).scalar_type() == at::kInt && compressed_type == at::kBFloat16) {
            return callBf16CSACompressorFwd(
                kv.data_ptr(), kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), score.data_ptr(),
                score_dims.data(), static_cast<std::uint8_t>(score_dims.size()), ape.data_ptr(), ape_dims.data(),
                static_cast<std::uint8_t>(ape_dims.size()), cu_seqlens.data_ptr(), cu_seqlens_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_dims.size()), cu_seqlens_comp.data_ptr(),
                cu_seqlens_comp_dims.data(), static_cast<std::uint8_t>(cu_seqlens_comp_dims.size()), compressed_data,
                compressed_dims.data(), static_cast<std::uint8_t>(compressed_dims.size()), ratio, head_dim, coff,
                total_comp, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.csa_compressor_fwd failed with status ",
                static_cast<unsigned int>(status));
    return compressed;
}

std::vector<at::Tensor> denseAttnScore(const at::Tensor& q, const at::Tensor& k, const at::Tensor& lse,
                                       std::int64_t ratio, double softmax_scale) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(k, "k");
    checkDeviceTensor(lse, "lse");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && lse.dim() == 3,
                "dense_attn_score expects q (B,Sq,Hq,D), k (B,Sk,1,D), lse (B,Sq,Hq)");
    at::Tensor out                   = torch::empty({q.size(0), q.size(1), k.size(1)}, q.options().dtype(at::kFloat));
    at::Tensor denom                 = torch::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    const auto q_dims                = q.sizes().vec();
    const auto k_dims                = k.sizes().vec();
    const auto lse_dims              = lse.sizes().vec();
    const auto out_dims              = out.sizes().vec();
    const auto denom_dims            = denom.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (lse).scalar_type() == at::kFloat && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return queryBf16DenseAttnScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, &required);
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return queryFp16DenseAttnScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, &required);
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return queryFp32DenseAttnScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, &required);
        }
        if ((lse).scalar_type() == at::kFloat && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return queryMixedFp32DenseAttnScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), lse.data_ptr(),
                lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, q.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (lse).scalar_type() == at::kFloat && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return callBf16DenseAttnScore(q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()),
                                          k.data_ptr(), k_dims.data(), static_cast<std::uint8_t>(k_dims.size()),
                                          lse.data_ptr(), lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()),
                                          out.data_ptr(), out_dims.data(), static_cast<std::uint8_t>(out_dims.size()),
                                          denom.data_ptr(), denom_dims.data(),
                                          static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale,
                                          (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return callFp16DenseAttnScore(q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()),
                                          k.data_ptr(), k_dims.data(), static_cast<std::uint8_t>(k_dims.size()),
                                          lse.data_ptr(), lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()),
                                          out.data_ptr(), out_dims.data(), static_cast<std::uint8_t>(out_dims.size()),
                                          denom.data_ptr(), denom_dims.data(),
                                          static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale,
                                          (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return callFp32DenseAttnScore(q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()),
                                          k.data_ptr(), k_dims.data(), static_cast<std::uint8_t>(k_dims.size()),
                                          lse.data_ptr(), lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()),
                                          out.data_ptr(), out_dims.data(), static_cast<std::uint8_t>(out_dims.size()),
                                          denom.data_ptr(), denom_dims.data(),
                                          static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale,
                                          (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((lse).scalar_type() == at::kFloat && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return callMixedFp32DenseAttnScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), lse.data_ptr(),
                lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.dense_attn_score failed with status ",
                static_cast<unsigned int>(status));
    return {out, denom};
}

std::vector<at::Tensor> denseAttnScoreThd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& lse,
                                          const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_kv,
                                          std::int64_t ratio, double softmax_scale, std::int64_t max_seqlen_kv,
                                          const std::optional<at::Tensor>& q_causal_offsets) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(k, "k");
    checkDeviceTensor(lse, "lse");
    checkDeviceTensor(cu_seqlens_q, "cu_seqlens_q");
    checkDeviceTensor(cu_seqlens_kv, "cu_seqlens_kv");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && lse.dim() == 2 && max_seqlen_kv > 0,
                "dense_attn_score_thd expects packed q/k/lse and max_seqlen_kv > 0");
    if (q_causal_offsets.has_value()) { checkDeviceTensor(*q_causal_offsets, "q_causal_offsets"); }
    at::Tensor out                    = torch::empty({q.size(0), max_seqlen_kv}, q.options().dtype(at::kFloat));
    at::Tensor denom                  = torch::empty({q.size(0)}, q.options().dtype(at::kFloat));
    const auto q_dims                 = q.sizes().vec();
    const auto k_dims                 = k.sizes().vec();
    const auto lse_dims               = lse.sizes().vec();
    const auto cu_seqlens_q_dims      = cu_seqlens_q.sizes().vec();
    const auto cu_seqlens_kv_dims     = cu_seqlens_kv.sizes().vec();
    const auto q_causal_offsets_dims  = (q_causal_offsets.has_value())
                                            ? (*q_causal_offsets).sizes().vec()
                                            : std::vector<std::int64_t>{cu_seqlens_q.size(0) - 1};
    const void* q_causal_offsets_data = (q_causal_offsets.has_value()) ? (*q_causal_offsets).data_ptr() : nullptr;
    const auto q_causal_offsets_type  = (q_causal_offsets.has_value()) ? (*q_causal_offsets).scalar_type() : at::kInt;
    const auto out_dims               = out.sizes().vec();
    const auto denom_dims             = denom.sizes().vec();
    std::uint64_t required            = 0;
    const unsigned char query_status  = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_kv).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return queryBf16DenseAttnScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, max_seqlen_kv, &required);
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return queryFp16DenseAttnScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, max_seqlen_kv, &required);
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return queryFp32DenseAttnScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, max_seqlen_kv, &required);
        }
        if ((lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_kv).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return queryMixedFp32DenseAttnScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), lse.data_ptr(),
                lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), cu_seqlens_q.data_ptr(),
                cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, max_seqlen_kv, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, q.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_kv).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return callBf16DenseAttnScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, max_seqlen_kv,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return callFp16DenseAttnScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, max_seqlen_kv,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return callFp32DenseAttnScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, max_seqlen_kv,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_kv).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return callMixedFp32DenseAttnScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), lse.data_ptr(),
                lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), cu_seqlens_q.data_ptr(),
                cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, softmax_scale, max_seqlen_kv,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.dense_attn_score_thd failed with status ",
                static_cast<unsigned int>(status));
    return {out, denom};
}

std::vector<at::Tensor> denseIndexerScore(const at::Tensor& q, const at::Tensor& k, const at::Tensor& w,
                                          std::int64_t ratio, double sm_scale) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(k, "k");
    checkDeviceTensor(w, "w");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && w.dim() == 3,
                "dense_indexer_score expects q (B,Sq,Hq,D), k (B,Sk,1,D), w (B,Sq,Hq)");
    at::Tensor out                   = torch::empty({q.size(0), q.size(1), k.size(1)}, q.options().dtype(at::kFloat));
    at::Tensor denom                 = torch::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    const auto q_dims                = q.sizes().vec();
    const auto k_dims                = k.sizes().vec();
    const auto w_dims                = w.sizes().vec();
    const auto out_dims              = out.sizes().vec();
    const auto denom_dims            = denom.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return queryBf16DenseIndexerScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, &required);
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return queryFp16DenseIndexerScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, &required);
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return queryFp32DenseIndexerScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, &required);
        }
        if ((out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return queryMixedFp32DenseIndexerScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, q.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return callBf16DenseIndexerScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return callFp16DenseIndexerScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return callFp32DenseIndexerScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return callMixedFp32DenseIndexerScore(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.dense_indexer_score failed with status ",
                static_cast<unsigned int>(status));
    return {out, denom};
}

std::vector<at::Tensor> denseIndexerScoreThd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& w,
                                             const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_kv,
                                             std::int64_t ratio, double sm_scale, std::int64_t max_seqlen_kv,
                                             const std::optional<at::Tensor>& q_causal_offsets) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(k, "k");
    checkDeviceTensor(w, "w");
    checkDeviceTensor(cu_seqlens_q, "cu_seqlens_q");
    checkDeviceTensor(cu_seqlens_kv, "cu_seqlens_kv");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && w.dim() == 2 && max_seqlen_kv > 0,
                "dense_indexer_score_thd expects packed q/k/w and max_seqlen_kv > 0");
    if (q_causal_offsets.has_value()) { checkDeviceTensor(*q_causal_offsets, "q_causal_offsets"); }

    at::Tensor out                    = torch::empty({q.size(0), max_seqlen_kv}, q.options().dtype(at::kFloat));
    at::Tensor denom                  = torch::empty({q.size(0)}, q.options().dtype(at::kFloat));
    const auto q_dims                 = q.sizes().vec();
    const auto k_dims                 = k.sizes().vec();
    const auto w_dims                 = w.sizes().vec();
    const auto cu_seqlens_q_dims      = cu_seqlens_q.sizes().vec();
    const auto cu_seqlens_kv_dims     = cu_seqlens_kv.sizes().vec();
    const auto q_causal_offsets_dims  = (q_causal_offsets.has_value())
                                            ? (*q_causal_offsets).sizes().vec()
                                            : std::vector<std::int64_t>{cu_seqlens_q.size(0) - 1};
    const void* q_causal_offsets_data = (q_causal_offsets.has_value()) ? (*q_causal_offsets).data_ptr() : nullptr;
    const auto q_causal_offsets_type  = (q_causal_offsets.has_value()) ? (*q_causal_offsets).scalar_type() : at::kInt;
    const auto out_dims               = out.sizes().vec();
    const auto denom_dims             = denom.sizes().vec();
    std::uint64_t required            = 0;
    const unsigned char query_status  = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_kv).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return queryBf16DenseIndexerScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, max_seqlen_kv, &required);
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return queryFp16DenseIndexerScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, max_seqlen_kv, &required);
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return queryFp32DenseIndexerScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, max_seqlen_kv, &required);
        }
        if ((cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return queryMixedFp32DenseIndexerScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), cu_seqlens_q.data_ptr(),
                cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, max_seqlen_kv, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, q.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_kv).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (out).scalar_type() == at::kFloat && (denom).scalar_type() == at::kFloat) {
            return callBf16DenseIndexerScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, max_seqlen_kv,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return callFp16DenseIndexerScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, max_seqlen_kv,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return callFp32DenseIndexerScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, max_seqlen_kv,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (denom).scalar_type() == at::kFloat) {
            return callMixedFp32DenseIndexerScoreThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), cu_seqlens_q.data_ptr(),
                cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()), out.data_ptr(),
                out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), denom.data_ptr(), denom_dims.data(),
                static_cast<std::uint8_t>(denom_dims.size()), ratio, sm_scale, max_seqlen_kv,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.dense_indexer_score_thd failed with status ",
                static_cast<unsigned int>(status));
    return {out, denom};
}

std::vector<at::Tensor> hAggregateBwd(const at::Tensor& grad_aggregated, const at::Tensor& x, const at::Tensor& h_pre) {
    checkDeviceTensor(grad_aggregated, "grad_aggregated");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(
        c10::impl::VirtualGuardImpl(grad_aggregated.device().type()).getDevice() == grad_aggregated.device()
            ? std::nullopt
            : std::optional<c10::Device>(grad_aggregated.device()));
    checkDeviceTensor(x, "x");
    checkDeviceTensor(h_pre, "h_pre");
    TORCH_CHECK(x.dim() == 4 && h_pre.dim() == 3 && grad_aggregated.dim() == 3,
                "h_aggregate_bwd expects grad (S,B,C), x (S,B,N,C), h_pre (S,B,N)");
    at::Tensor grad_x               = torch::empty_like(x);
    at::Tensor grad_h               = torch::empty_like(h_pre);
    const auto grad_aggregated_dims = grad_aggregated.sizes().vec();
    TORCH_CHECK(x.device() == grad_aggregated.device(), "all operands must be on the same device");
    const auto x_dims = x.sizes().vec();
    TORCH_CHECK(h_pre.device() == grad_aggregated.device(), "all operands must be on the same device");
    const auto h_pre_dims = h_pre.sizes().vec();
    TORCH_CHECK(grad_x.device() == grad_aggregated.device(), "all operands must be on the same device");
    const auto grad_x_dims = grad_x.sizes().vec();
    TORCH_CHECK(grad_h.device() == grad_aggregated.device(), "all operands must be on the same device");
    const auto grad_h_pre_dims = grad_h.sizes().vec();
    const unsigned char status = [&]() -> unsigned char {
        if ((grad_aggregated).scalar_type() == at::kFloat && (x).scalar_type() == at::kFloat &&
            (h_pre).scalar_type() == at::kFloat && (grad_x).scalar_type() == at::kFloat &&
            (grad_h).scalar_type() == at::kFloat) {
            return callFp32HAggregateBwd(
                grad_aggregated.data_ptr(), grad_aggregated_dims.data(),
                static_cast<std::uint8_t>(grad_aggregated_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_h.data_ptr(), grad_h_pre_dims.data(),
                static_cast<std::uint8_t>(grad_h_pre_dims.size()), currentStream());
        }
        if ((h_pre).scalar_type() == at::kFloat && (grad_h).scalar_type() == at::kFloat) {
            return callMixedFp32HAggregateBwd(
                grad_aggregated.data_ptr(), grad_aggregated_dims.data(),
                static_cast<std::uint8_t>(grad_aggregated_dims.size()), numericTypeCode(grad_aggregated), x.data_ptr(),
                x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), numericTypeCode(x), h_pre.data_ptr(),
                h_pre_dims.data(), static_cast<std::uint8_t>(h_pre_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), numericTypeCode(grad_x), grad_h.data_ptr(),
                grad_h_pre_dims.data(), static_cast<std::uint8_t>(grad_h_pre_dims.size()), currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.h_aggregate_bwd failed with status ",
                static_cast<unsigned int>(status));
    return {grad_x, grad_h};
}

at::Tensor hAggregateFwd(const at::Tensor& x, const at::Tensor& h_pre) {
    checkDeviceTensor(x, "x");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(x.device().type()).getDevice() == x.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(x.device()));
    checkDeviceTensor(h_pre, "h_pre");
    TORCH_CHECK(x.dim() == 4 && h_pre.dim() == 3, "h_aggregate_fwd expects x (S,B,N,C) and h_pre (S,B,N)");
    at::Tensor aggregated = torch::empty({x.size(0), x.size(1), x.size(3)}, x.options());
    const auto x_dims     = x.sizes().vec();
    TORCH_CHECK(h_pre.device() == x.device(), "all operands must be on the same device");
    const auto h_pre_dims = h_pre.sizes().vec();
    TORCH_CHECK(aggregated.device() == x.device(), "all operands must be on the same device");
    const auto aggregated_dims = aggregated.sizes().vec();
    const unsigned char status = [&]() -> unsigned char {
        if ((x).scalar_type() == at::kFloat && (h_pre).scalar_type() == at::kFloat &&
            (aggregated).scalar_type() == at::kFloat) {
            return callFp32HAggregateFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), h_pre.data_ptr(),
                h_pre_dims.data(), static_cast<std::uint8_t>(h_pre_dims.size()), aggregated.data_ptr(),
                aggregated_dims.data(), static_cast<std::uint8_t>(aggregated_dims.size()), currentStream());
        }
        if ((h_pre).scalar_type() == at::kFloat) {
            return callMixedFp32HAggregateFwd(x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()),
                                              numericTypeCode(x), h_pre.data_ptr(), h_pre_dims.data(),
                                              static_cast<std::uint8_t>(h_pre_dims.size()), aggregated.data_ptr(),
                                              aggregated_dims.data(), static_cast<std::uint8_t>(aggregated_dims.size()),
                                              numericTypeCode(aggregated), currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.h_aggregate_fwd failed with status ",
                static_cast<unsigned int>(status));
    return aggregated;
}

std::vector<at::Tensor> hPostBdaBwd(const at::Tensor& grad_output, const at::Tensor& residual, const at::Tensor& x,
                                    const at::Tensor& h_res, const at::Tensor& h_post,
                                    const std::optional<at::Tensor>& bias, bool want_bias_gradient) {
    checkDeviceTensor(grad_output, "grad_output");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(grad_output.device().type()).getDevice() ==
                                                        grad_output.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(grad_output.device()));
    checkDeviceTensor(residual, "residual");
    checkDeviceTensor(x, "x");
    checkDeviceTensor(h_res, "h_res");
    checkDeviceTensor(h_post, "h_post");
    if (bias.has_value()) { checkDeviceTensor(*bias, "bias"); }
    TORCH_CHECK(
        h_res.dim() == 4 && residual.dim() == 4 && h_post.dim() == 3 && x.dim() == 3,
        "h_post_bda_bwd expects grad (S,B,N,C), residual (S,B,N,C), x (S,B,C), h_res (S,B,N,N), " "h_post (S,B,N)");
    const std::int64_t channels = residual.size(3);
    at::Tensor grad_residual    = torch::empty_like(residual);
    at::Tensor grad_x           = torch::empty_like(x);
    at::Tensor grad_h_res       = torch::empty_like(h_res);
    at::Tensor grad_h_post      = torch::empty_like(h_post);
    at::Tensor grad_bias        = torch::empty({channels}, residual.options().dtype(at::kFloat));
    if (!want_bias_gradient) { grad_bias = at::Tensor(); }
    const auto grad_output_dims = grad_output.sizes().vec();
    TORCH_CHECK(residual.device() == grad_output.device(), "all operands must be on the same device");
    const auto residual_dims = residual.sizes().vec();
    TORCH_CHECK(x.device() == grad_output.device(), "all operands must be on the same device");
    const auto x_dims = x.sizes().vec();
    if (bias.has_value()) {
        TORCH_CHECK((*bias).device() == grad_output.device(), "all operands must be on the same device");
    }
    const auto bias_dims  = (bias.has_value()) ? (*bias).sizes().vec() : std::vector<std::int64_t>{channels};
    const void* bias_data = (bias.has_value()) ? (*bias).data_ptr() : nullptr;
    const auto bias_type  = (bias.has_value()) ? (*bias).scalar_type() : at::kFloat;
    TORCH_CHECK(h_res.device() == grad_output.device(), "all operands must be on the same device");
    const auto h_res_dims = h_res.sizes().vec();
    TORCH_CHECK(h_post.device() == grad_output.device(), "all operands must be on the same device");
    const auto h_post_dims = h_post.sizes().vec();
    TORCH_CHECK(grad_residual.device() == grad_output.device(), "all operands must be on the same device");
    const auto grad_residual_dims = grad_residual.sizes().vec();
    TORCH_CHECK(grad_x.device() == grad_output.device(), "all operands must be on the same device");
    const auto grad_x_dims = grad_x.sizes().vec();
    TORCH_CHECK(grad_h_res.device() == grad_output.device(), "all operands must be on the same device");
    const auto grad_h_res_dims = grad_h_res.sizes().vec();
    TORCH_CHECK(grad_h_post.device() == grad_output.device(), "all operands must be on the same device");
    const auto grad_h_post_dims = grad_h_post.sizes().vec();
    if (want_bias_gradient) {
        TORCH_CHECK(grad_bias.device() == grad_output.device(), "all operands must be on the same device");
    }
    const auto grad_bias_dims  = (want_bias_gradient) ? grad_bias.sizes().vec() : std::vector<std::int64_t>{channels};
    const void* grad_bias_data = (want_bias_gradient) ? grad_bias.data_ptr() : nullptr;
    const auto grad_bias_type  = (want_bias_gradient) ? (grad_bias).scalar_type() : at::kFloat;
    const unsigned char status = [&]() -> unsigned char {
        if ((grad_output).scalar_type() == at::kBFloat16 && (residual).scalar_type() == at::kBFloat16 &&
            (x).scalar_type() == at::kBFloat16 && bias_type == at::kFloat && (h_res).scalar_type() == at::kBFloat16 &&
            (h_post).scalar_type() == at::kBFloat16 && (grad_residual).scalar_type() == at::kBFloat16 &&
            (grad_x).scalar_type() == at::kBFloat16 && (grad_h_res).scalar_type() == at::kBFloat16 &&
            (grad_h_post).scalar_type() == at::kBFloat16 && grad_bias_type == at::kFloat) {
            return callBf16HPostBdaBwd(
                grad_output.data_ptr(), grad_output_dims.data(), static_cast<std::uint8_t>(grad_output_dims.size()),
                residual.data_ptr(), residual_dims.data(), static_cast<std::uint8_t>(residual_dims.size()),
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), bias_data, bias_dims.data(),
                static_cast<std::uint8_t>(bias_dims.size()), h_res.data_ptr(), h_res_dims.data(),
                static_cast<std::uint8_t>(h_res_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), grad_residual.data_ptr(), grad_residual_dims.data(),
                static_cast<std::uint8_t>(grad_residual_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_h_res.data_ptr(), grad_h_res_dims.data(),
                static_cast<std::uint8_t>(grad_h_res_dims.size()), grad_h_post.data_ptr(), grad_h_post_dims.data(),
                static_cast<std::uint8_t>(grad_h_post_dims.size()), grad_bias_data, grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), currentStream());
        }
        if ((grad_output).scalar_type() == at::kHalf && (residual).scalar_type() == at::kHalf &&
            (x).scalar_type() == at::kHalf && bias_type == at::kFloat && (h_res).scalar_type() == at::kHalf &&
            (h_post).scalar_type() == at::kHalf && (grad_residual).scalar_type() == at::kHalf &&
            (grad_x).scalar_type() == at::kHalf && (grad_h_res).scalar_type() == at::kHalf &&
            (grad_h_post).scalar_type() == at::kHalf && grad_bias_type == at::kFloat) {
            return callFp16HPostBdaBwd(
                grad_output.data_ptr(), grad_output_dims.data(), static_cast<std::uint8_t>(grad_output_dims.size()),
                residual.data_ptr(), residual_dims.data(), static_cast<std::uint8_t>(residual_dims.size()),
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), bias_data, bias_dims.data(),
                static_cast<std::uint8_t>(bias_dims.size()), h_res.data_ptr(), h_res_dims.data(),
                static_cast<std::uint8_t>(h_res_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), grad_residual.data_ptr(), grad_residual_dims.data(),
                static_cast<std::uint8_t>(grad_residual_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_h_res.data_ptr(), grad_h_res_dims.data(),
                static_cast<std::uint8_t>(grad_h_res_dims.size()), grad_h_post.data_ptr(), grad_h_post_dims.data(),
                static_cast<std::uint8_t>(grad_h_post_dims.size()), grad_bias_data, grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), currentStream());
        }
        if ((grad_output).scalar_type() == at::kFloat && (residual).scalar_type() == at::kFloat &&
            (x).scalar_type() == at::kFloat && bias_type == at::kFloat && (h_res).scalar_type() == at::kFloat &&
            (h_post).scalar_type() == at::kFloat && (grad_residual).scalar_type() == at::kFloat &&
            (grad_x).scalar_type() == at::kFloat && (grad_h_res).scalar_type() == at::kFloat &&
            (grad_h_post).scalar_type() == at::kFloat && grad_bias_type == at::kFloat) {
            return callFp32HPostBdaBwd(
                grad_output.data_ptr(), grad_output_dims.data(), static_cast<std::uint8_t>(grad_output_dims.size()),
                residual.data_ptr(), residual_dims.data(), static_cast<std::uint8_t>(residual_dims.size()),
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), bias_data, bias_dims.data(),
                static_cast<std::uint8_t>(bias_dims.size()), h_res.data_ptr(), h_res_dims.data(),
                static_cast<std::uint8_t>(h_res_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), grad_residual.data_ptr(), grad_residual_dims.data(),
                static_cast<std::uint8_t>(grad_residual_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_h_res.data_ptr(), grad_h_res_dims.data(),
                static_cast<std::uint8_t>(grad_h_res_dims.size()), grad_h_post.data_ptr(), grad_h_post_dims.data(),
                static_cast<std::uint8_t>(grad_h_post_dims.size()), grad_bias_data, grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.h_post_bda_bwd failed with status ", static_cast<unsigned int>(status));
    return {grad_residual, grad_x, grad_h_res, grad_h_post, grad_bias};
}

at::Tensor hPostBdaFwd(const at::Tensor& h_res, const at::Tensor& residual, const at::Tensor& h_post,
                       const at::Tensor& x, const std::optional<at::Tensor>& bias) {
    checkDeviceTensor(h_res, "h_res");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(h_res.device().type()).getDevice() ==
                                                        h_res.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(h_res.device()));
    checkDeviceTensor(residual, "residual");
    checkDeviceTensor(h_post, "h_post");
    checkDeviceTensor(x, "x");
    if (bias.has_value()) { checkDeviceTensor(*bias, "bias"); }
    TORCH_CHECK(h_res.dim() == 4 && residual.dim() == 4 && h_post.dim() == 3 && x.dim() == 3,
                "h_post_bda_fwd expects h_res (S,B,N,N), residual (S,B,N,C), h_post (S,B,N), x (S,B,C)");
    at::Tensor output     = torch::empty_like(residual);
    const auto h_res_dims = h_res.sizes().vec();
    TORCH_CHECK(residual.device() == h_res.device(), "all operands must be on the same device");
    const auto residual_dims = residual.sizes().vec();
    TORCH_CHECK(h_post.device() == h_res.device(), "all operands must be on the same device");
    const auto h_post_dims = h_post.sizes().vec();
    TORCH_CHECK(x.device() == h_res.device(), "all operands must be on the same device");
    const auto x_dims = x.sizes().vec();
    if (bias.has_value()) {
        TORCH_CHECK((*bias).device() == h_res.device(), "all operands must be on the same device");
    }
    const auto bias_dims  = (bias.has_value()) ? (*bias).sizes().vec() : std::vector<std::int64_t>{residual.size(3)};
    const void* bias_data = (bias.has_value()) ? (*bias).data_ptr() : nullptr;
    const auto bias_type  = (bias.has_value()) ? (*bias).scalar_type() : at::kFloat;
    TORCH_CHECK(output.device() == h_res.device(), "all operands must be on the same device");
    const auto output_dims     = output.sizes().vec();
    const unsigned char status = [&]() -> unsigned char {
        if ((h_res).scalar_type() == at::kBFloat16 && (residual).scalar_type() == at::kBFloat16 &&
            (h_post).scalar_type() == at::kBFloat16 && (x).scalar_type() == at::kBFloat16 && bias_type == at::kFloat &&
            (output).scalar_type() == at::kBFloat16) {
            return callBf16HPostBdaFwd(
                h_res.data_ptr(), h_res_dims.data(), static_cast<std::uint8_t>(h_res_dims.size()), residual.data_ptr(),
                residual_dims.data(), static_cast<std::uint8_t>(residual_dims.size()), h_post.data_ptr(),
                h_post_dims.data(), static_cast<std::uint8_t>(h_post_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), bias_data, bias_dims.data(),
                static_cast<std::uint8_t>(bias_dims.size()), output.data_ptr(), output_dims.data(),
                static_cast<std::uint8_t>(output_dims.size()), currentStream());
        }
        if ((h_res).scalar_type() == at::kHalf && (residual).scalar_type() == at::kHalf &&
            (h_post).scalar_type() == at::kHalf && (x).scalar_type() == at::kHalf && bias_type == at::kFloat &&
            (output).scalar_type() == at::kHalf) {
            return callFp16HPostBdaFwd(
                h_res.data_ptr(), h_res_dims.data(), static_cast<std::uint8_t>(h_res_dims.size()), residual.data_ptr(),
                residual_dims.data(), static_cast<std::uint8_t>(residual_dims.size()), h_post.data_ptr(),
                h_post_dims.data(), static_cast<std::uint8_t>(h_post_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), bias_data, bias_dims.data(),
                static_cast<std::uint8_t>(bias_dims.size()), output.data_ptr(), output_dims.data(),
                static_cast<std::uint8_t>(output_dims.size()), currentStream());
        }
        if ((h_res).scalar_type() == at::kFloat && (residual).scalar_type() == at::kFloat &&
            (h_post).scalar_type() == at::kFloat && (x).scalar_type() == at::kFloat && bias_type == at::kFloat &&
            (output).scalar_type() == at::kFloat) {
            return callFp32HPostBdaFwd(
                h_res.data_ptr(), h_res_dims.data(), static_cast<std::uint8_t>(h_res_dims.size()), residual.data_ptr(),
                residual_dims.data(), static_cast<std::uint8_t>(residual_dims.size()), h_post.data_ptr(),
                h_post_dims.data(), static_cast<std::uint8_t>(h_post_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), bias_data, bias_dims.data(),
                static_cast<std::uint8_t>(bias_dims.size()), output.data_ptr(), output_dims.data(),
                static_cast<std::uint8_t>(output_dims.size()), currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.h_post_bda_fwd failed with status ", static_cast<unsigned int>(status));
    return output;
}

std::vector<at::Tensor> indexerBwdDense(const at::Tensor& index_q, const at::Tensor& weights, const at::Tensor& index_k,
                                        const at::Tensor& attn_score, const at::Tensor& attn_l1norm,
                                        const at::Tensor& index_score, const at::Tensor& index_lse, double sm_scale,
                                        double loss_coeff, double grad_loss, std::int64_t ratio) {
    checkDeviceTensor(index_q, "index_q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(index_q.device().type()).getDevice() ==
                                                        index_q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(index_q.device()));
    checkDeviceTensor(weights, "weights");
    checkDeviceTensor(index_k, "index_k");
    checkDeviceTensor(attn_score, "attn_score");
    checkDeviceTensor(attn_l1norm, "attn_l1norm");
    checkDeviceTensor(index_score, "index_score");
    checkDeviceTensor(index_lse, "index_lse");
    TORCH_CHECK(index_q.dim() == 4 && weights.dim() == 3 && index_k.dim() == 3 && attn_score.dim() == 3 &&
                    index_lse.dim() == 2,
                "indexer_bwd_dense expects index_q (B,Sq,H,D), weights (B,Sq,H), index_k (B,Sk,D), "
                "attn_score/index_score (B,Sq,Sk), attn_l1norm (B,Sq), index_lse (B,Sq)");
    at::Tensor dq                    = torch::empty_like(index_q);
    at::Tensor dw                    = torch::empty_like(weights);
    at::Tensor dk                    = torch::empty_like(index_k);
    const auto index_q_dims          = index_q.sizes().vec();
    const auto weights_dims          = weights.sizes().vec();
    const auto index_k_dims          = index_k.sizes().vec();
    const auto attn_score_dims       = attn_score.sizes().vec();
    const auto attn_l1norm_dims      = attn_l1norm.sizes().vec();
    const auto index_score_dims      = index_score.sizes().vec();
    const auto index_lse_dims        = index_lse.sizes().vec();
    const auto grad_index_score_dims = index_score.sizes().vec();
    const auto dq_dims               = dq.sizes().vec();
    const auto dw_dims               = dw.sizes().vec();
    const auto dk_dims               = dk.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((index_q).scalar_type() == at::kBFloat16 && (weights).scalar_type() == at::kBFloat16 &&
            (index_k).scalar_type() == at::kBFloat16 && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (dq).scalar_type() == at::kBFloat16 && (dw).scalar_type() == at::kBFloat16 &&
            (dk).scalar_type() == at::kBFloat16) {
            return queryBf16IndexerBwdDense(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                index_score.data_ptr(), grad_index_score_dims.data(),
                static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio, &required);
        }
        if ((index_q).scalar_type() == at::kHalf && (weights).scalar_type() == at::kHalf &&
            (index_k).scalar_type() == at::kHalf && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (dq).scalar_type() == at::kHalf && (dw).scalar_type() == at::kHalf && (dk).scalar_type() == at::kHalf) {
            return queryFp16IndexerBwdDense(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                index_score.data_ptr(), grad_index_score_dims.data(),
                static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio, &required);
        }
        if ((index_q).scalar_type() == at::kFloat && (weights).scalar_type() == at::kFloat &&
            (index_k).scalar_type() == at::kFloat && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (dq).scalar_type() == at::kFloat && (dw).scalar_type() == at::kFloat && (dk).scalar_type() == at::kFloat) {
            return queryFp32IndexerBwdDense(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                index_score.data_ptr(), grad_index_score_dims.data(),
                static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio, &required);
        }
        if ((attn_score).scalar_type() == at::kFloat && (attn_l1norm).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (index_lse).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat) {
            return queryMixedFp32IndexerBwdDense(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                numericTypeCode(index_q), weights.data_ptr(), weights_dims.data(),
                static_cast<std::uint8_t>(weights_dims.size()), numericTypeCode(weights), index_k.data_ptr(),
                index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()), numericTypeCode(index_k),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                index_score.data_ptr(), grad_index_score_dims.data(),
                static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), numericTypeCode(dw), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), numericTypeCode(dk), sm_scale, loss_coeff, grad_loss, ratio,
                &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, dq.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((index_q).scalar_type() == at::kBFloat16 && (weights).scalar_type() == at::kBFloat16 &&
            (index_k).scalar_type() == at::kBFloat16 && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (dq).scalar_type() == at::kBFloat16 && (dw).scalar_type() == at::kBFloat16 &&
            (dk).scalar_type() == at::kBFloat16) {
            return callBf16IndexerBwdDense(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                index_score.data_ptr(), grad_index_score_dims.data(),
                static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((index_q).scalar_type() == at::kHalf && (weights).scalar_type() == at::kHalf &&
            (index_k).scalar_type() == at::kHalf && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (dq).scalar_type() == at::kHalf && (dw).scalar_type() == at::kHalf && (dk).scalar_type() == at::kHalf) {
            return callFp16IndexerBwdDense(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                index_score.data_ptr(), grad_index_score_dims.data(),
                static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((index_q).scalar_type() == at::kFloat && (weights).scalar_type() == at::kFloat &&
            (index_k).scalar_type() == at::kFloat && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (dq).scalar_type() == at::kFloat && (dw).scalar_type() == at::kFloat && (dk).scalar_type() == at::kFloat) {
            return callFp32IndexerBwdDense(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                index_score.data_ptr(), grad_index_score_dims.data(),
                static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((attn_score).scalar_type() == at::kFloat && (attn_l1norm).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (index_lse).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat) {
            return callMixedFp32IndexerBwdDense(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                numericTypeCode(index_q), weights.data_ptr(), weights_dims.data(),
                static_cast<std::uint8_t>(weights_dims.size()), numericTypeCode(weights), index_k.data_ptr(),
                index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()), numericTypeCode(index_k),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                index_score.data_ptr(), grad_index_score_dims.data(),
                static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), numericTypeCode(dw), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), numericTypeCode(dk), sm_scale, loss_coeff, grad_loss, ratio,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.indexer_bwd_dense failed with status ",
                static_cast<unsigned int>(status));
    return {index_score, dq, dw, dk};
}

std::vector<at::Tensor> indexerBwdDenseThd(const at::Tensor& index_q, const at::Tensor& weights,
                                           const at::Tensor& index_k, const at::Tensor& attn_score,
                                           const at::Tensor& attn_l1norm, const at::Tensor& index_score,
                                           const at::Tensor& index_lse, const at::Tensor& cu_seqlens_q,
                                           const at::Tensor& cu_seqlens_k, double sm_scale, double loss_coeff,
                                           double grad_loss, std::int64_t ratio,
                                           const std::optional<at::Tensor>& q_causal_offsets) {
    checkDeviceTensor(index_q, "index_q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(index_q.device().type()).getDevice() ==
                                                        index_q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(index_q.device()));
    checkDeviceTensor(weights, "weights");
    checkDeviceTensor(index_k, "index_k");
    checkDeviceTensor(attn_score, "attn_score");
    checkDeviceTensor(attn_l1norm, "attn_l1norm");
    checkDeviceTensor(index_score, "index_score");
    checkDeviceTensor(index_lse, "index_lse");
    checkDeviceTensor(cu_seqlens_q, "cu_seqlens_q");
    checkDeviceTensor(cu_seqlens_k, "cu_seqlens_k");
    if (q_causal_offsets.has_value()) { checkDeviceTensor(*q_causal_offsets, "q_causal_offsets"); }
    TORCH_CHECK(index_q.dim() == 3 && weights.dim() == 2 && index_k.dim() == 2 && attn_score.dim() == 2 &&
                    index_lse.dim() == 1,
                "indexer_bwd_dense_thd expects packed index_q (total_q,H,D), weights (total_q,H), "
                "index_k (total_k,D), attn_score/index_score (total_q,max_sk), index_lse (total_q)");
    at::Tensor dq                     = torch::empty_like(index_q);
    at::Tensor dw                     = torch::empty_like(weights);
    at::Tensor dk                     = torch::empty_like(index_k);
    const auto index_q_dims           = index_q.sizes().vec();
    const auto weights_dims           = weights.sizes().vec();
    const auto index_k_dims           = index_k.sizes().vec();
    const auto attn_score_dims        = attn_score.sizes().vec();
    const auto attn_l1norm_dims       = attn_l1norm.sizes().vec();
    const auto index_score_dims       = index_score.sizes().vec();
    const auto index_lse_dims         = index_lse.sizes().vec();
    const auto cu_seqlens_q_dims      = cu_seqlens_q.sizes().vec();
    const auto cu_seqlens_k_dims      = cu_seqlens_k.sizes().vec();
    const auto q_causal_offsets_dims  = (q_causal_offsets.has_value())
                                            ? (*q_causal_offsets).sizes().vec()
                                            : std::vector<std::int64_t>{cu_seqlens_q.size(0) - 1};
    const void* q_causal_offsets_data = (q_causal_offsets.has_value()) ? (*q_causal_offsets).data_ptr() : nullptr;
    const auto q_causal_offsets_type  = (q_causal_offsets.has_value()) ? (*q_causal_offsets).scalar_type() : at::kInt;
    const auto grad_index_score_dims  = index_score.sizes().vec();
    const auto dq_dims                = dq.sizes().vec();
    const auto dw_dims                = dw.sizes().vec();
    const auto dk_dims                = dk.sizes().vec();
    std::uint64_t required            = 0;
    const unsigned char query_status  = [&]() -> unsigned char {
        if ((index_q).scalar_type() == at::kBFloat16 && (weights).scalar_type() == at::kBFloat16 &&
            (index_k).scalar_type() == at::kBFloat16 && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_k).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (index_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kBFloat16 &&
            (dw).scalar_type() == at::kBFloat16 && (dk).scalar_type() == at::kBFloat16) {
            return queryBf16IndexerBwdDenseThd(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()),
                cu_seqlens_k.data_ptr(), cu_seqlens_k_dims.data(), static_cast<std::uint8_t>(cu_seqlens_k_dims.size()),
                q_causal_offsets_data, q_causal_offsets_dims.data(),
                static_cast<std::uint8_t>(q_causal_offsets_dims.size()), index_score.data_ptr(),
                grad_index_score_dims.data(), static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio, &required);
        }
        if ((index_q).scalar_type() == at::kHalf && (weights).scalar_type() == at::kHalf &&
            (index_k).scalar_type() == at::kHalf && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_k).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (index_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kHalf &&
            (dw).scalar_type() == at::kHalf && (dk).scalar_type() == at::kHalf) {
            return queryFp16IndexerBwdDenseThd(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()),
                cu_seqlens_k.data_ptr(), cu_seqlens_k_dims.data(), static_cast<std::uint8_t>(cu_seqlens_k_dims.size()),
                q_causal_offsets_data, q_causal_offsets_dims.data(),
                static_cast<std::uint8_t>(q_causal_offsets_dims.size()), index_score.data_ptr(),
                grad_index_score_dims.data(), static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio, &required);
        }
        if ((index_q).scalar_type() == at::kFloat && (weights).scalar_type() == at::kFloat &&
            (index_k).scalar_type() == at::kFloat && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_k).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (index_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kFloat &&
            (dw).scalar_type() == at::kFloat && (dk).scalar_type() == at::kFloat) {
            return queryFp32IndexerBwdDenseThd(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()),
                cu_seqlens_k.data_ptr(), cu_seqlens_k_dims.data(), static_cast<std::uint8_t>(cu_seqlens_k_dims.size()),
                q_causal_offsets_data, q_causal_offsets_dims.data(),
                static_cast<std::uint8_t>(q_causal_offsets_dims.size()), index_score.data_ptr(),
                grad_index_score_dims.data(), static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio, &required);
        }
        if ((attn_score).scalar_type() == at::kFloat && (attn_l1norm).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (index_lse).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_k).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (index_score).scalar_type() == at::kFloat) {
            return queryMixedFp32IndexerBwdDenseThd(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                numericTypeCode(index_q), weights.data_ptr(), weights_dims.data(),
                static_cast<std::uint8_t>(weights_dims.size()), numericTypeCode(weights), index_k.data_ptr(),
                index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()), numericTypeCode(index_k),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()),
                cu_seqlens_k.data_ptr(), cu_seqlens_k_dims.data(), static_cast<std::uint8_t>(cu_seqlens_k_dims.size()),
                q_causal_offsets_data, q_causal_offsets_dims.data(),
                static_cast<std::uint8_t>(q_causal_offsets_dims.size()), index_score.data_ptr(),
                grad_index_score_dims.data(), static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dw.data_ptr(),
                dw_dims.data(), static_cast<std::uint8_t>(dw_dims.size()), numericTypeCode(dw), dk.data_ptr(),
                dk_dims.data(), static_cast<std::uint8_t>(dk_dims.size()), numericTypeCode(dk), sm_scale, loss_coeff,
                grad_loss, ratio, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, dq.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((index_q).scalar_type() == at::kBFloat16 && (weights).scalar_type() == at::kBFloat16 &&
            (index_k).scalar_type() == at::kBFloat16 && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_k).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (index_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kBFloat16 &&
            (dw).scalar_type() == at::kBFloat16 && (dk).scalar_type() == at::kBFloat16) {
            return callBf16IndexerBwdDenseThd(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()),
                cu_seqlens_k.data_ptr(), cu_seqlens_k_dims.data(), static_cast<std::uint8_t>(cu_seqlens_k_dims.size()),
                q_causal_offsets_data, q_causal_offsets_dims.data(),
                static_cast<std::uint8_t>(q_causal_offsets_dims.size()), index_score.data_ptr(),
                grad_index_score_dims.data(), static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((index_q).scalar_type() == at::kHalf && (weights).scalar_type() == at::kHalf &&
            (index_k).scalar_type() == at::kHalf && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_k).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (index_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kHalf &&
            (dw).scalar_type() == at::kHalf && (dk).scalar_type() == at::kHalf) {
            return callFp16IndexerBwdDenseThd(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()),
                cu_seqlens_k.data_ptr(), cu_seqlens_k_dims.data(), static_cast<std::uint8_t>(cu_seqlens_k_dims.size()),
                q_causal_offsets_data, q_causal_offsets_dims.data(),
                static_cast<std::uint8_t>(q_causal_offsets_dims.size()), index_score.data_ptr(),
                grad_index_score_dims.data(), static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((index_q).scalar_type() == at::kFloat && (weights).scalar_type() == at::kFloat &&
            (index_k).scalar_type() == at::kFloat && (attn_score).scalar_type() == at::kFloat &&
            (attn_l1norm).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (index_lse).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_k).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (index_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kFloat &&
            (dw).scalar_type() == at::kFloat && (dk).scalar_type() == at::kFloat) {
            return callFp32IndexerBwdDenseThd(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()),
                cu_seqlens_k.data_ptr(), cu_seqlens_k_dims.data(), static_cast<std::uint8_t>(cu_seqlens_k_dims.size()),
                q_causal_offsets_data, q_causal_offsets_dims.data(),
                static_cast<std::uint8_t>(q_causal_offsets_dims.size()), index_score.data_ptr(),
                grad_index_score_dims.data(), static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss, ratio,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((attn_score).scalar_type() == at::kFloat && (attn_l1norm).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (index_lse).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_k).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (index_score).scalar_type() == at::kFloat) {
            return callMixedFp32IndexerBwdDenseThd(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                numericTypeCode(index_q), weights.data_ptr(), weights_dims.data(),
                static_cast<std::uint8_t>(weights_dims.size()), numericTypeCode(weights), index_k.data_ptr(),
                index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()), numericTypeCode(index_k),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                attn_l1norm.data_ptr(), attn_l1norm_dims.data(), static_cast<std::uint8_t>(attn_l1norm_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                index_lse.data_ptr(), index_lse_dims.data(), static_cast<std::uint8_t>(index_lse_dims.size()),
                cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()),
                cu_seqlens_k.data_ptr(), cu_seqlens_k_dims.data(), static_cast<std::uint8_t>(cu_seqlens_k_dims.size()),
                q_causal_offsets_data, q_causal_offsets_dims.data(),
                static_cast<std::uint8_t>(q_causal_offsets_dims.size()), index_score.data_ptr(),
                grad_index_score_dims.data(), static_cast<std::uint8_t>(grad_index_score_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dw.data_ptr(),
                dw_dims.data(), static_cast<std::uint8_t>(dw_dims.size()), numericTypeCode(dw), dk.data_ptr(),
                dk_dims.data(), static_cast<std::uint8_t>(dk_dims.size()), numericTypeCode(dk), sm_scale, loss_coeff,
                grad_loss, ratio, (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.indexer_bwd_dense_thd failed with status ",
                static_cast<unsigned int>(status));
    return {index_score, dq, dw, dk};
}

std::vector<at::Tensor> indexerBwdSparse(const at::Tensor& index_q, const at::Tensor& weights,
                                         const at::Tensor& index_k, const at::Tensor& attn_score,
                                         const at::Tensor& index_score, const at::Tensor& topk_indices, double sm_scale,
                                         double loss_coeff, const py::object& grad_loss, bool topk_indices_global) {
    checkDeviceTensor(index_q, "index_q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(index_q.device().type()).getDevice() ==
                                                        index_q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(index_q.device()));
    checkDeviceTensor(weights, "weights");
    checkDeviceTensor(index_k, "index_k");
    checkDeviceTensor(attn_score, "attn_score");
    checkDeviceTensor(index_score, "index_score");
    checkDeviceTensor(topk_indices, "topk_indices");
    TORCH_CHECK(index_q.dim() == 4 && weights.dim() == 3 && index_k.dim() == 3 && attn_score.dim() == 3,
                "indexer_bwd_sparse expects index_q (B,Sq,H,D), weights (B,Sq,H), index_k (B,Sk,D), "
                "attn_score/index_score/topk_idxs (B,Sq,topk)");
    // Keep GPU scale tensors on the stream: converting one to double calls item()
    // and synchronizes all preceding work before this backward can be submitted.
    double grad_loss_value     = 1.0;
    const void* grad_loss_data = nullptr;
    at::Tensor grad_loss_tensor;
    if (THPVariable_Check(grad_loss.ptr())) {
        grad_loss_tensor = grad_loss.cast<at::Tensor>();
        TORCH_CHECK(grad_loss_tensor.numel() == 1, "grad_loss must contain one element");
        if (grad_loss_tensor.is_cuda() && grad_loss_tensor.scalar_type() == at::kFloat &&
            index_q.scalar_type() == at::kBFloat16 && weights.scalar_type() == at::kBFloat16 &&
            index_k.scalar_type() == at::kBFloat16) {
            TORCH_CHECK(grad_loss_tensor.device() == index_q.device(), "grad_loss must be on the query device");
            grad_loss_data = grad_loss_tensor.data_ptr();
        } else {
            grad_loss_value = grad_loss_tensor.item<double>();
        }
    } else {
        grad_loss_value = grad_loss.cast<double>();
    }
    at::Tensor dq                    = torch::empty_like(index_q);
    at::Tensor dw                    = torch::empty_like(weights);
    at::Tensor dk                    = torch::empty_like(index_k);
    const auto index_q_dims          = index_q.sizes().vec();
    const auto weights_dims          = weights.sizes().vec();
    const auto index_k_dims          = index_k.sizes().vec();
    const auto attn_score_dims       = attn_score.sizes().vec();
    const auto index_score_dims      = index_score.sizes().vec();
    const auto topk_indices_dims     = topk_indices.sizes().vec();
    const auto grad_attn_score_dims  = attn_score.sizes().vec();
    const auto dq_dims               = dq.sizes().vec();
    const auto dw_dims               = dw.sizes().vec();
    const auto dk_dims               = dk.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((index_q).scalar_type() == at::kBFloat16 && (weights).scalar_type() == at::kBFloat16 &&
            (index_k).scalar_type() == at::kBFloat16 && (attn_score).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (attn_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kBFloat16 &&
            (dw).scalar_type() == at::kBFloat16 && (dk).scalar_type() == at::kBFloat16) {
            return queryBf16IndexerBwdSparse(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                attn_score.data_ptr(), grad_attn_score_dims.data(),
                static_cast<std::uint8_t>(grad_attn_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss_value, topk_indices_global,
                &required, grad_loss_data);
        }
        if ((index_q).scalar_type() == at::kHalf && (weights).scalar_type() == at::kHalf &&
            (index_k).scalar_type() == at::kHalf && (attn_score).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (attn_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kHalf &&
            (dw).scalar_type() == at::kHalf && (dk).scalar_type() == at::kHalf) {
            return queryFp16IndexerBwdSparse(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                attn_score.data_ptr(), grad_attn_score_dims.data(),
                static_cast<std::uint8_t>(grad_attn_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss_value, topk_indices_global,
                &required);
        }
        if ((index_q).scalar_type() == at::kFloat && (weights).scalar_type() == at::kFloat &&
            (index_k).scalar_type() == at::kFloat && (attn_score).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (attn_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kFloat &&
            (dw).scalar_type() == at::kFloat && (dk).scalar_type() == at::kFloat) {
            return queryFp32IndexerBwdSparse(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                attn_score.data_ptr(), grad_attn_score_dims.data(),
                static_cast<std::uint8_t>(grad_attn_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss_value, topk_indices_global,
                &required);
        }
        if ((attn_score).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && (attn_score).scalar_type() == at::kFloat) {
            return queryMixedFp32IndexerBwdSparse(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                numericTypeCode(index_q), weights.data_ptr(), weights_dims.data(),
                static_cast<std::uint8_t>(weights_dims.size()), numericTypeCode(weights), index_k.data_ptr(),
                index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()), numericTypeCode(index_k),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                attn_score.data_ptr(), grad_attn_score_dims.data(),
                static_cast<std::uint8_t>(grad_attn_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), numericTypeCode(dw), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), numericTypeCode(dk), sm_scale, loss_coeff, grad_loss_value,
                topk_indices_global, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, dq.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((index_q).scalar_type() == at::kBFloat16 && (weights).scalar_type() == at::kBFloat16 &&
            (index_k).scalar_type() == at::kBFloat16 && (attn_score).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (attn_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kBFloat16 &&
            (dw).scalar_type() == at::kBFloat16 && (dk).scalar_type() == at::kBFloat16) {
            return callBf16IndexerBwdSparse(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                attn_score.data_ptr(), grad_attn_score_dims.data(),
                static_cast<std::uint8_t>(grad_attn_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss_value, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream(), grad_loss_data);
        }
        if ((index_q).scalar_type() == at::kHalf && (weights).scalar_type() == at::kHalf &&
            (index_k).scalar_type() == at::kHalf && (attn_score).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (attn_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kHalf &&
            (dw).scalar_type() == at::kHalf && (dk).scalar_type() == at::kHalf) {
            return callFp16IndexerBwdSparse(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                attn_score.data_ptr(), grad_attn_score_dims.data(),
                static_cast<std::uint8_t>(grad_attn_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss_value, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((index_q).scalar_type() == at::kFloat && (weights).scalar_type() == at::kFloat &&
            (index_k).scalar_type() == at::kFloat && (attn_score).scalar_type() == at::kFloat &&
            (index_score).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (attn_score).scalar_type() == at::kFloat && (dq).scalar_type() == at::kFloat &&
            (dw).scalar_type() == at::kFloat && (dk).scalar_type() == at::kFloat) {
            return callFp32IndexerBwdSparse(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                weights.data_ptr(), weights_dims.data(), static_cast<std::uint8_t>(weights_dims.size()),
                index_k.data_ptr(), index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                attn_score.data_ptr(), grad_attn_score_dims.data(),
                static_cast<std::uint8_t>(grad_attn_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), sm_scale, loss_coeff, grad_loss_value, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((attn_score).scalar_type() == at::kFloat && (index_score).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && (attn_score).scalar_type() == at::kFloat) {
            return callMixedFp32IndexerBwdSparse(
                index_q.data_ptr(), index_q_dims.data(), static_cast<std::uint8_t>(index_q_dims.size()),
                numericTypeCode(index_q), weights.data_ptr(), weights_dims.data(),
                static_cast<std::uint8_t>(weights_dims.size()), numericTypeCode(weights), index_k.data_ptr(),
                index_k_dims.data(), static_cast<std::uint8_t>(index_k_dims.size()), numericTypeCode(index_k),
                attn_score.data_ptr(), attn_score_dims.data(), static_cast<std::uint8_t>(attn_score_dims.size()),
                index_score.data_ptr(), index_score_dims.data(), static_cast<std::uint8_t>(index_score_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                attn_score.data_ptr(), grad_attn_score_dims.data(),
                static_cast<std::uint8_t>(grad_attn_score_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dw.data_ptr(), dw_dims.data(),
                static_cast<std::uint8_t>(dw_dims.size()), numericTypeCode(dw), dk.data_ptr(), dk_dims.data(),
                static_cast<std::uint8_t>(dk_dims.size()), numericTypeCode(dk), sm_scale, loss_coeff, grad_loss_value,
                topk_indices_global, (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.indexer_bwd_sparse failed with status ",
                static_cast<unsigned int>(status));
    return {attn_score, dq, dw, dk};
}

at::Tensor indexerFwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& w, std::int64_t ratio,
                      double sm_scale) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(k, "k");
    checkDeviceTensor(w, "w");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && w.dim() == 3,
                "indexer_fwd expects q (B,Sq,Hq,D), k (B,Sk,1,D), w (B,Sq,Hq)");
    at::Tensor scores                = torch::empty({q.size(0), q.size(1), k.size(1)}, q.options().dtype(at::kFloat));
    const auto q_dims                = q.sizes().vec();
    const auto k_dims                = k.sizes().vec();
    const auto w_dims                = w.sizes().vec();
    const auto scores_dims           = scores.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (scores).scalar_type() == at::kFloat) {
            return queryBf16IndexerFwd(q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()),
                                       k.data_ptr(), k_dims.data(), static_cast<std::uint8_t>(k_dims.size()),
                                       w.data_ptr(), w_dims.data(), static_cast<std::uint8_t>(w_dims.size()),
                                       scores.data_ptr(), scores_dims.data(),
                                       static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale, &required);
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (scores).scalar_type() == at::kFloat) {
            return queryFp16IndexerFwd(q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()),
                                       k.data_ptr(), k_dims.data(), static_cast<std::uint8_t>(k_dims.size()),
                                       w.data_ptr(), w_dims.data(), static_cast<std::uint8_t>(w_dims.size()),
                                       scores.data_ptr(), scores_dims.data(),
                                       static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale, &required);
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (scores).scalar_type() == at::kFloat) {
            return queryFp32IndexerFwd(q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()),
                                       k.data_ptr(), k_dims.data(), static_cast<std::uint8_t>(k_dims.size()),
                                       w.data_ptr(), w_dims.data(), static_cast<std::uint8_t>(w_dims.size()),
                                       scores.data_ptr(), scores_dims.data(),
                                       static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale, &required);
        }
        if ((scores).scalar_type() == at::kFloat) {
            return queryMixedFp32IndexerFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), scores.data_ptr(),
                scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, q.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (scores).scalar_type() == at::kFloat) {
            return callBf16IndexerFwd(q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()),
                                      k.data_ptr(), k_dims.data(), static_cast<std::uint8_t>(k_dims.size()),
                                      w.data_ptr(), w_dims.data(), static_cast<std::uint8_t>(w_dims.size()),
                                      scores.data_ptr(), scores_dims.data(),
                                      static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                                      (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (scores).scalar_type() == at::kFloat) {
            return callFp16IndexerFwd(q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()),
                                      k.data_ptr(), k_dims.data(), static_cast<std::uint8_t>(k_dims.size()),
                                      w.data_ptr(), w_dims.data(), static_cast<std::uint8_t>(w_dims.size()),
                                      scores.data_ptr(), scores_dims.data(),
                                      static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                                      (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (scores).scalar_type() == at::kFloat) {
            return callFp32IndexerFwd(q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()),
                                      k.data_ptr(), k_dims.data(), static_cast<std::uint8_t>(k_dims.size()),
                                      w.data_ptr(), w_dims.data(), static_cast<std::uint8_t>(w_dims.size()),
                                      scores.data_ptr(), scores_dims.data(),
                                      static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                                      (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((scores).scalar_type() == at::kFloat) {
            return callMixedFp32IndexerFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), scores.data_ptr(),
                scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.indexer_fwd failed with status ", static_cast<unsigned int>(status));
    return scores;
}

at::Tensor indexerFwdThd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& w, const at::Tensor& cu_seqlens_q,
                         const at::Tensor& cu_seqlens_kv, std::int64_t ratio, double sm_scale,
                         std::int64_t max_seqlen_kv, const std::optional<at::Tensor>& q_causal_offsets) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(k, "k");
    checkDeviceTensor(w, "w");
    checkDeviceTensor(cu_seqlens_q, "cu_seqlens_q");
    checkDeviceTensor(cu_seqlens_kv, "cu_seqlens_kv");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && w.dim() == 2 && max_seqlen_kv > 0,
                "indexer_fwd_thd expects packed q (total_q,Hq,D), k (total_k,1,D), w (total_q,Hq)");
    if (q_causal_offsets.has_value()) { checkDeviceTensor(*q_causal_offsets, "q_causal_offsets"); }

    at::Tensor scores                 = torch::empty({q.size(0), max_seqlen_kv}, q.options().dtype(at::kFloat));
    const auto q_dims                 = q.sizes().vec();
    const auto k_dims                 = k.sizes().vec();
    const auto w_dims                 = w.sizes().vec();
    const auto cu_seqlens_q_dims      = cu_seqlens_q.sizes().vec();
    const auto cu_seqlens_kv_dims     = cu_seqlens_kv.sizes().vec();
    const auto q_causal_offsets_dims  = (q_causal_offsets.has_value())
                                            ? (*q_causal_offsets).sizes().vec()
                                            : std::vector<std::int64_t>{cu_seqlens_q.size(0) - 1};
    const void* q_causal_offsets_data = (q_causal_offsets.has_value()) ? (*q_causal_offsets).data_ptr() : nullptr;
    const auto q_causal_offsets_type  = (q_causal_offsets.has_value()) ? (*q_causal_offsets).scalar_type() : at::kInt;
    const auto scores_dims            = scores.sizes().vec();
    std::uint64_t required            = 0;
    const unsigned char query_status  = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_kv).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (scores).scalar_type() == at::kFloat) {
            return queryBf16IndexerFwdThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()),
                scores.data_ptr(), scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                max_seqlen_kv, &required);
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (scores).scalar_type() == at::kFloat) {
            return queryFp16IndexerFwdThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()),
                scores.data_ptr(), scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                max_seqlen_kv, &required);
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (scores).scalar_type() == at::kFloat) {
            return queryFp32IndexerFwdThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()),
                scores.data_ptr(), scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                max_seqlen_kv, &required);
        }
        if ((cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (scores).scalar_type() == at::kFloat) {
            return queryMixedFp32IndexerFwdThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), cu_seqlens_q.data_ptr(),
                cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()),
                scores.data_ptr(), scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                max_seqlen_kv, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, q.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_kv).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (scores).scalar_type() == at::kFloat) {
            return callBf16IndexerFwdThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()),
                scores.data_ptr(), scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                max_seqlen_kv, (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (scores).scalar_type() == at::kFloat) {
            return callFp16IndexerFwdThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()),
                scores.data_ptr(), scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                max_seqlen_kv, (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (scores).scalar_type() == at::kFloat) {
            return callFp32IndexerFwdThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()),
                scores.data_ptr(), scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                max_seqlen_kv, (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((cu_seqlens_q).scalar_type() == at::kInt && (cu_seqlens_kv).scalar_type() == at::kInt &&
            q_causal_offsets_type == at::kInt && (scores).scalar_type() == at::kFloat) {
            return callMixedFp32IndexerFwdThd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), cu_seqlens_q.data_ptr(),
                cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()), cu_seqlens_kv.data_ptr(),
                cu_seqlens_kv_dims.data(), static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()),
                scores.data_ptr(), scores_dims.data(), static_cast<std::uint8_t>(scores_dims.size()), ratio, sm_scale,
                max_seqlen_kv, (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.indexer_fwd_thd failed with status ",
                static_cast<unsigned int>(status));
    return scores;
}

std::vector<at::Tensor> indexerTopK(const at::Tensor& input_values, const at::Tensor& seq_lens, std::int64_t top_k,
                                    std::int64_t next_n, bool order_values) {
    checkDeviceTensor(input_values, "input_values");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(input_values.device().type()).getDevice() ==
                                                        input_values.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(input_values.device()));
    checkDeviceTensor(seq_lens, "seq_lens");
    TORCH_CHECK(input_values.dim() == 2 && seq_lens.dim() == 1, "indexer_topk expects (n_rows,n_cols) and (rows,)");
    at::Tensor indices = torch::empty({input_values.size(0), top_k}, input_values.options().dtype(at::kInt));
    at::Tensor values  = order_values
                             ? torch::empty({input_values.size(0), top_k}, input_values.options().dtype(at::kFloat))
                             : at::Tensor();
    const auto input_values_dims = input_values.sizes().vec();
    const auto seq_lens_dims     = seq_lens.sizes().vec();
    const auto out_indices_dims  = indices.sizes().vec();
    const auto out_values_dims =
        (order_values) ? values.sizes().vec() : std::vector<std::int64_t>{input_values.size(0), top_k};
    const void* out_values_data = (order_values) ? values.data_ptr() : nullptr;
    const auto out_values_type  = (order_values) ? (values).scalar_type() : at::kFloat;
    const unsigned char status  = [&]() -> unsigned char {
        if ((input_values).scalar_type() == at::kFloat && (seq_lens).scalar_type() == at::kInt &&
            (indices).scalar_type() == at::kInt && out_values_type == at::kFloat) {
            return callFp32IndexerTopK(
                input_values.data_ptr(), input_values_dims.data(), static_cast<std::uint8_t>(input_values_dims.size()),
                seq_lens.data_ptr(), seq_lens_dims.data(), static_cast<std::uint8_t>(seq_lens_dims.size()),
                indices.data_ptr(), out_indices_dims.data(), static_cast<std::uint8_t>(out_indices_dims.size()),
                out_values_data, out_values_dims.data(), static_cast<std::uint8_t>(out_values_dims.size()), top_k,
                next_n, order_values, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.indexer_topk failed with status ", static_cast<unsigned int>(status));
    return {indices, values};
}

std::vector<at::Tensor> indexerTopKThd(const at::Tensor& input_values, const at::Tensor& cu_seqlens_q,
                                       const at::Tensor& cu_seqlens_kv, std::int64_t ratio, std::int64_t top_k,
                                       bool order_values, const std::optional<at::Tensor>& q_causal_offsets) {
    checkDeviceTensor(input_values, "input_values");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(input_values.device().type()).getDevice() ==
                                                        input_values.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(input_values.device()));
    checkDeviceTensor(cu_seqlens_q, "cu_seqlens_q");
    checkDeviceTensor(cu_seqlens_kv, "cu_seqlens_kv");
    if (q_causal_offsets.has_value()) { checkDeviceTensor(*q_causal_offsets, "q_causal_offsets"); }
    TORCH_CHECK(input_values.dim() == 2 && cu_seqlens_q.dim() == 1 && cu_seqlens_kv.dim() == 1,
                "indexer_topk_thd expects (n_rows,n_cols) values and rank-one cumulative lengths");
    at::Tensor indices = torch::empty({input_values.size(0), top_k}, input_values.options().dtype(at::kInt));
    at::Tensor values  = order_values
                             ? torch::empty({input_values.size(0), top_k}, input_values.options().dtype(at::kFloat))
                             : at::Tensor();
    const auto input_values_dims      = input_values.sizes().vec();
    const auto cu_seqlens_q_dims      = cu_seqlens_q.sizes().vec();
    const auto cu_seqlens_kv_dims     = cu_seqlens_kv.sizes().vec();
    const auto q_causal_offsets_dims  = (q_causal_offsets.has_value())
                                            ? (*q_causal_offsets).sizes().vec()
                                            : std::vector<std::int64_t>{cu_seqlens_q.size(0) - 1};
    const void* q_causal_offsets_data = (q_causal_offsets.has_value()) ? (*q_causal_offsets).data_ptr() : nullptr;
    const auto q_causal_offsets_type  = (q_causal_offsets.has_value()) ? (*q_causal_offsets).scalar_type() : at::kInt;
    const auto out_indices_dims       = indices.sizes().vec();
    const auto out_values_dims =
        (order_values) ? values.sizes().vec() : std::vector<std::int64_t>{input_values.size(0), top_k};
    const void* out_values_data = (order_values) ? values.data_ptr() : nullptr;
    const auto out_values_type  = (order_values) ? (values).scalar_type() : at::kFloat;
    const unsigned char status  = [&]() -> unsigned char {
        if ((input_values).scalar_type() == at::kFloat && (cu_seqlens_q).scalar_type() == at::kInt &&
            (cu_seqlens_kv).scalar_type() == at::kInt && q_causal_offsets_type == at::kInt &&
            (indices).scalar_type() == at::kInt && out_values_type == at::kFloat) {
            return callFp32IndexerTopKThd(
                input_values.data_ptr(), input_values_dims.data(), static_cast<std::uint8_t>(input_values_dims.size()),
                cu_seqlens_q.data_ptr(), cu_seqlens_q_dims.data(), static_cast<std::uint8_t>(cu_seqlens_q_dims.size()),
                cu_seqlens_kv.data_ptr(), cu_seqlens_kv_dims.data(),
                static_cast<std::uint8_t>(cu_seqlens_kv_dims.size()), q_causal_offsets_data,
                q_causal_offsets_dims.data(), static_cast<std::uint8_t>(q_causal_offsets_dims.size()),
                indices.data_ptr(), out_indices_dims.data(), static_cast<std::uint8_t>(out_indices_dims.size()),
                out_values_data, out_values_dims.data(), static_cast<std::uint8_t>(out_values_dims.size()), ratio,
                top_k, order_values, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.indexer_topk_thd failed with status ",
                static_cast<unsigned int>(status));
    return {indices, values};
}

std::vector<at::Tensor> projRmsBwd(const at::Tensor& grad_proj, const at::Tensor& grad_r, const at::Tensor& x,
                                   const at::Tensor& weight, const at::Tensor& r, double eps) {
    checkDeviceTensor(grad_proj, "grad_proj");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(grad_proj.device().type()).getDevice() ==
                                                        grad_proj.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(grad_proj.device()));
    checkDeviceTensor(grad_r, "grad_r");
    checkDeviceTensor(x, "x");
    checkDeviceTensor(weight, "weight");
    checkDeviceTensor(r, "r");
    TORCH_CHECK(grad_proj.dim() == 2 && grad_r.dim() == 2 && x.dim() == 2 && weight.dim() == 2 && r.dim() == 2,
                "proj_rms_bwd expects grad_proj (M,N), grad_r (M,1), x (M,K), weight (N,K), r (M,1)");
    at::Tensor grad_x         = torch::empty_like(x);
    at::Tensor grad_weight    = torch::empty_like(weight);
    const auto grad_proj_dims = grad_proj.sizes().vec();
    TORCH_CHECK(grad_r.device() == grad_proj.device(), "all operands must be on the same device");
    const auto grad_r_dims = grad_r.sizes().vec();
    TORCH_CHECK(x.device() == grad_proj.device(), "all operands must be on the same device");
    const auto x_dims = x.sizes().vec();
    TORCH_CHECK(weight.device() == grad_proj.device(), "all operands must be on the same device");
    const auto weight_dims = weight.sizes().vec();
    TORCH_CHECK(r.device() == grad_proj.device(), "all operands must be on the same device");
    const auto r_dims = r.sizes().vec();
    TORCH_CHECK(grad_x.device() == grad_proj.device(), "all operands must be on the same device");
    const auto grad_x_dims = grad_x.sizes().vec();
    TORCH_CHECK(grad_weight.device() == grad_proj.device(), "all operands must be on the same device");
    const auto grad_weight_dims = grad_weight.sizes().vec();
    const unsigned char status  = [&]() -> unsigned char {
        if ((grad_proj).scalar_type() == at::kBFloat16 && (grad_r).scalar_type() == at::kBFloat16 &&
            (x).scalar_type() == at::kBFloat16 && (weight).scalar_type() == at::kBFloat16 &&
            (r).scalar_type() == at::kBFloat16 && (grad_x).scalar_type() == at::kBFloat16 &&
            (grad_weight).scalar_type() == at::kBFloat16) {
            return callBf16ProjRmsBwd(
                grad_proj.data_ptr(), grad_proj_dims.data(), static_cast<std::uint8_t>(grad_proj_dims.size()),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()), x.data_ptr(),
                x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_weight.data_ptr(), grad_weight_dims.data(),
                static_cast<std::uint8_t>(grad_weight_dims.size()), eps, currentStream());
        }
        if ((grad_proj).scalar_type() == at::kHalf && (grad_r).scalar_type() == at::kHalf &&
            (x).scalar_type() == at::kHalf && (weight).scalar_type() == at::kHalf && (r).scalar_type() == at::kHalf &&
            (grad_x).scalar_type() == at::kHalf && (grad_weight).scalar_type() == at::kHalf) {
            return callFp16ProjRmsBwd(
                grad_proj.data_ptr(), grad_proj_dims.data(), static_cast<std::uint8_t>(grad_proj_dims.size()),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()), x.data_ptr(),
                x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_weight.data_ptr(), grad_weight_dims.data(),
                static_cast<std::uint8_t>(grad_weight_dims.size()), eps, currentStream());
        }
        if ((grad_proj).scalar_type() == at::kFloat && (grad_r).scalar_type() == at::kFloat &&
            (x).scalar_type() == at::kFloat && (weight).scalar_type() == at::kFloat &&
            (r).scalar_type() == at::kFloat && (grad_x).scalar_type() == at::kFloat &&
            (grad_weight).scalar_type() == at::kFloat) {
            return callFp32ProjRmsBwd(
                grad_proj.data_ptr(), grad_proj_dims.data(), static_cast<std::uint8_t>(grad_proj_dims.size()),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()), x.data_ptr(),
                x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_weight.data_ptr(), grad_weight_dims.data(),
                static_cast<std::uint8_t>(grad_weight_dims.size()), eps, currentStream());
        }
        {
            return callMixedFp32ProjRmsBwd(
                grad_proj.data_ptr(), grad_proj_dims.data(), static_cast<std::uint8_t>(grad_proj_dims.size()),
                numericTypeCode(grad_proj), grad_r.data_ptr(), grad_r_dims.data(),
                static_cast<std::uint8_t>(grad_r_dims.size()), numericTypeCode(grad_r), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), numericTypeCode(x), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), numericTypeCode(weight), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), numericTypeCode(r), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), numericTypeCode(grad_x), grad_weight.data_ptr(),
                grad_weight_dims.data(), static_cast<std::uint8_t>(grad_weight_dims.size()),
                numericTypeCode(grad_weight), eps, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.proj_rms_bwd failed with status ", static_cast<unsigned int>(status));
    return {grad_x, grad_weight};
}

std::vector<at::Tensor> projRmsComputeHBwd(const at::Tensor& grad_h_pre, const at::Tensor& grad_h_post,
                                           const at::Tensor& grad_h_res, const at::Tensor& grad_r,
                                           const at::Tensor& h_pre, const at::Tensor& h_post, const at::Tensor& proj,
                                           const at::Tensor& r, const at::Tensor& x, const at::Tensor& weight,
                                           const at::Tensor& alpha_pre, const at::Tensor& alpha_post,
                                           const at::Tensor& alpha_res, std::int64_t n, double eps,
                                           double compute_h_eps) {
    checkDeviceTensor(grad_h_pre, "grad_h_pre");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(grad_h_pre.device().type()).getDevice() ==
                                                        grad_h_pre.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(grad_h_pre.device()));
    checkDeviceTensor(grad_h_post, "grad_h_post");
    checkDeviceTensor(grad_h_res, "grad_h_res");
    checkDeviceTensor(grad_r, "grad_r");
    checkDeviceTensor(h_pre, "h_pre");
    checkDeviceTensor(h_post, "h_post");
    checkDeviceTensor(proj, "proj");
    checkDeviceTensor(r, "r");
    checkDeviceTensor(x, "x");
    checkDeviceTensor(weight, "weight");
    checkDeviceTensor(alpha_pre, "alpha_pre");
    checkDeviceTensor(alpha_post, "alpha_post");
    checkDeviceTensor(alpha_res, "alpha_res");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "proj_rms_compute_h_bwd expects x (M,K) and weight (N,K)");
    at::Tensor grad_x          = torch::empty_like(x);
    at::Tensor grad_weight     = torch::empty_like(weight);
    at::Tensor grad_alpha      = torch::empty({3}, alpha_pre.options().dtype(at::kFloat));
    at::Tensor grad_bias       = torch::empty({weight.size(0)}, x.options().dtype(at::kFloat));
    const auto grad_h_pre_dims = grad_h_pre.sizes().vec();
    TORCH_CHECK(grad_h_post.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto grad_h_post_dims = grad_h_post.sizes().vec();
    TORCH_CHECK(grad_h_res.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto grad_h_res_dims = grad_h_res.sizes().vec();
    TORCH_CHECK(grad_r.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto grad_r_dims = grad_r.sizes().vec();
    TORCH_CHECK(h_pre.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto h_pre_dims = h_pre.sizes().vec();
    TORCH_CHECK(h_post.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto h_post_dims = h_post.sizes().vec();
    TORCH_CHECK(proj.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto proj_dims = proj.sizes().vec();
    TORCH_CHECK(r.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto r_ch_dims = r.sizes().vec();
    TORCH_CHECK(x.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto x_dims = x.sizes().vec();
    TORCH_CHECK(weight.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto weight_dims = weight.sizes().vec();
    TORCH_CHECK(alpha_pre.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto alpha_pre_dims = alpha_pre.sizes().vec();
    TORCH_CHECK(alpha_post.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto alpha_post_dims = alpha_post.sizes().vec();
    TORCH_CHECK(alpha_res.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto alpha_res_dims = alpha_res.sizes().vec();
    TORCH_CHECK(grad_x.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto grad_x_dims = grad_x.sizes().vec();
    TORCH_CHECK(grad_weight.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto grad_weight_dims = grad_weight.sizes().vec();
    TORCH_CHECK(grad_alpha.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto grad_alpha_dims = grad_alpha.sizes().vec();
    TORCH_CHECK(grad_bias.device() == grad_h_pre.device(), "all operands must be on the same device");
    const auto grad_bias_dims        = grad_bias.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((grad_h_pre).scalar_type() == at::kBFloat16 && (grad_h_post).scalar_type() == at::kBFloat16 &&
            (grad_h_res).scalar_type() == at::kBFloat16 && (grad_r).scalar_type() == at::kBFloat16 &&
            (h_pre).scalar_type() == at::kBFloat16 && (h_post).scalar_type() == at::kBFloat16 &&
            (proj).scalar_type() == at::kBFloat16 && (r).scalar_type() == at::kBFloat16 &&
            (x).scalar_type() == at::kBFloat16 && (weight).scalar_type() == at::kBFloat16 &&
            (alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (grad_x).scalar_type() == at::kBFloat16 &&
            (grad_weight).scalar_type() == at::kBFloat16 && (grad_alpha).scalar_type() == at::kFloat &&
            (grad_bias).scalar_type() == at::kFloat) {
            return queryBf16ProjRmsComputeHBwd(
                grad_h_pre.data_ptr(), grad_h_pre_dims.data(), static_cast<std::uint8_t>(grad_h_pre_dims.size()),
                grad_h_post.data_ptr(), grad_h_post_dims.data(), static_cast<std::uint8_t>(grad_h_post_dims.size()),
                grad_h_res.data_ptr(), grad_h_res_dims.data(), static_cast<std::uint8_t>(grad_h_res_dims.size()),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()), h_pre.data_ptr(),
                h_pre_dims.data(), static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_ch_dims.data(),
                static_cast<std::uint8_t>(r_ch_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_weight.data_ptr(), grad_weight_dims.data(),
                static_cast<std::uint8_t>(grad_weight_dims.size()), grad_alpha.data_ptr(), grad_alpha_dims.data(),
                static_cast<std::uint8_t>(grad_alpha_dims.size()), grad_bias.data_ptr(), grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), n, eps, compute_h_eps, &required);
        }
        if ((grad_h_pre).scalar_type() == at::kHalf && (grad_h_post).scalar_type() == at::kHalf &&
            (grad_h_res).scalar_type() == at::kHalf && (grad_r).scalar_type() == at::kHalf &&
            (h_pre).scalar_type() == at::kHalf && (h_post).scalar_type() == at::kHalf &&
            (proj).scalar_type() == at::kHalf && (r).scalar_type() == at::kHalf && (x).scalar_type() == at::kHalf &&
            (weight).scalar_type() == at::kHalf && (alpha_pre).scalar_type() == at::kFloat &&
            (alpha_post).scalar_type() == at::kFloat && (alpha_res).scalar_type() == at::kFloat &&
            (grad_x).scalar_type() == at::kHalf && (grad_weight).scalar_type() == at::kHalf &&
            (grad_alpha).scalar_type() == at::kFloat && (grad_bias).scalar_type() == at::kFloat) {
            return queryFp16ProjRmsComputeHBwd(
                grad_h_pre.data_ptr(), grad_h_pre_dims.data(), static_cast<std::uint8_t>(grad_h_pre_dims.size()),
                grad_h_post.data_ptr(), grad_h_post_dims.data(), static_cast<std::uint8_t>(grad_h_post_dims.size()),
                grad_h_res.data_ptr(), grad_h_res_dims.data(), static_cast<std::uint8_t>(grad_h_res_dims.size()),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()), h_pre.data_ptr(),
                h_pre_dims.data(), static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_ch_dims.data(),
                static_cast<std::uint8_t>(r_ch_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_weight.data_ptr(), grad_weight_dims.data(),
                static_cast<std::uint8_t>(grad_weight_dims.size()), grad_alpha.data_ptr(), grad_alpha_dims.data(),
                static_cast<std::uint8_t>(grad_alpha_dims.size()), grad_bias.data_ptr(), grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), n, eps, compute_h_eps, &required);
        }
        if ((grad_h_pre).scalar_type() == at::kFloat && (grad_h_post).scalar_type() == at::kFloat &&
            (grad_h_res).scalar_type() == at::kFloat && (grad_r).scalar_type() == at::kFloat &&
            (h_pre).scalar_type() == at::kFloat && (h_post).scalar_type() == at::kFloat &&
            (proj).scalar_type() == at::kFloat && (r).scalar_type() == at::kFloat && (x).scalar_type() == at::kFloat &&
            (weight).scalar_type() == at::kFloat && (alpha_pre).scalar_type() == at::kFloat &&
            (alpha_post).scalar_type() == at::kFloat && (alpha_res).scalar_type() == at::kFloat &&
            (grad_x).scalar_type() == at::kFloat && (grad_weight).scalar_type() == at::kFloat &&
            (grad_alpha).scalar_type() == at::kFloat && (grad_bias).scalar_type() == at::kFloat) {
            return queryFp32ProjRmsComputeHBwd(
                grad_h_pre.data_ptr(), grad_h_pre_dims.data(), static_cast<std::uint8_t>(grad_h_pre_dims.size()),
                grad_h_post.data_ptr(), grad_h_post_dims.data(), static_cast<std::uint8_t>(grad_h_post_dims.size()),
                grad_h_res.data_ptr(), grad_h_res_dims.data(), static_cast<std::uint8_t>(grad_h_res_dims.size()),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()), h_pre.data_ptr(),
                h_pre_dims.data(), static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_ch_dims.data(),
                static_cast<std::uint8_t>(r_ch_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_weight.data_ptr(), grad_weight_dims.data(),
                static_cast<std::uint8_t>(grad_weight_dims.size()), grad_alpha.data_ptr(), grad_alpha_dims.data(),
                static_cast<std::uint8_t>(grad_alpha_dims.size()), grad_bias.data_ptr(), grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), n, eps, compute_h_eps, &required);
        }
        if ((alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (grad_alpha).scalar_type() == at::kFloat &&
            (grad_bias).scalar_type() == at::kFloat) {
            return queryMixedFp32ProjRmsComputeHBwd(
                grad_h_pre.data_ptr(), grad_h_pre_dims.data(), static_cast<std::uint8_t>(grad_h_pre_dims.size()),
                numericTypeCode(grad_h_pre), grad_h_post.data_ptr(), grad_h_post_dims.data(),
                static_cast<std::uint8_t>(grad_h_post_dims.size()), numericTypeCode(grad_h_post), grad_h_res.data_ptr(),
                grad_h_res_dims.data(), static_cast<std::uint8_t>(grad_h_res_dims.size()), numericTypeCode(grad_h_res),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()),
                numericTypeCode(grad_r), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), numericTypeCode(h_pre), h_post.data_ptr(),
                h_post_dims.data(), static_cast<std::uint8_t>(h_post_dims.size()), numericTypeCode(h_post),
                proj.data_ptr(), proj_dims.data(), static_cast<std::uint8_t>(proj_dims.size()), numericTypeCode(proj),
                r.data_ptr(), r_ch_dims.data(), static_cast<std::uint8_t>(r_ch_dims.size()), numericTypeCode(r),
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), numericTypeCode(x),
                weight.data_ptr(), weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()),
                numericTypeCode(weight), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), numericTypeCode(grad_x), grad_weight.data_ptr(),
                grad_weight_dims.data(), static_cast<std::uint8_t>(grad_weight_dims.size()),
                numericTypeCode(grad_weight), grad_alpha.data_ptr(), grad_alpha_dims.data(),
                static_cast<std::uint8_t>(grad_alpha_dims.size()), grad_bias.data_ptr(), grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), n, eps, compute_h_eps, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, grad_x.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((grad_h_pre).scalar_type() == at::kBFloat16 && (grad_h_post).scalar_type() == at::kBFloat16 &&
            (grad_h_res).scalar_type() == at::kBFloat16 && (grad_r).scalar_type() == at::kBFloat16 &&
            (h_pre).scalar_type() == at::kBFloat16 && (h_post).scalar_type() == at::kBFloat16 &&
            (proj).scalar_type() == at::kBFloat16 && (r).scalar_type() == at::kBFloat16 &&
            (x).scalar_type() == at::kBFloat16 && (weight).scalar_type() == at::kBFloat16 &&
            (alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (grad_x).scalar_type() == at::kBFloat16 &&
            (grad_weight).scalar_type() == at::kBFloat16 && (grad_alpha).scalar_type() == at::kFloat &&
            (grad_bias).scalar_type() == at::kFloat) {
            return callBf16ProjRmsComputeHBwd(
                grad_h_pre.data_ptr(), grad_h_pre_dims.data(), static_cast<std::uint8_t>(grad_h_pre_dims.size()),
                grad_h_post.data_ptr(), grad_h_post_dims.data(), static_cast<std::uint8_t>(grad_h_post_dims.size()),
                grad_h_res.data_ptr(), grad_h_res_dims.data(), static_cast<std::uint8_t>(grad_h_res_dims.size()),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()), h_pre.data_ptr(),
                h_pre_dims.data(), static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_ch_dims.data(),
                static_cast<std::uint8_t>(r_ch_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_weight.data_ptr(), grad_weight_dims.data(),
                static_cast<std::uint8_t>(grad_weight_dims.size()), grad_alpha.data_ptr(), grad_alpha_dims.data(),
                static_cast<std::uint8_t>(grad_alpha_dims.size()), grad_bias.data_ptr(), grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), n, eps, compute_h_eps,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((grad_h_pre).scalar_type() == at::kHalf && (grad_h_post).scalar_type() == at::kHalf &&
            (grad_h_res).scalar_type() == at::kHalf && (grad_r).scalar_type() == at::kHalf &&
            (h_pre).scalar_type() == at::kHalf && (h_post).scalar_type() == at::kHalf &&
            (proj).scalar_type() == at::kHalf && (r).scalar_type() == at::kHalf && (x).scalar_type() == at::kHalf &&
            (weight).scalar_type() == at::kHalf && (alpha_pre).scalar_type() == at::kFloat &&
            (alpha_post).scalar_type() == at::kFloat && (alpha_res).scalar_type() == at::kFloat &&
            (grad_x).scalar_type() == at::kHalf && (grad_weight).scalar_type() == at::kHalf &&
            (grad_alpha).scalar_type() == at::kFloat && (grad_bias).scalar_type() == at::kFloat) {
            return callFp16ProjRmsComputeHBwd(
                grad_h_pre.data_ptr(), grad_h_pre_dims.data(), static_cast<std::uint8_t>(grad_h_pre_dims.size()),
                grad_h_post.data_ptr(), grad_h_post_dims.data(), static_cast<std::uint8_t>(grad_h_post_dims.size()),
                grad_h_res.data_ptr(), grad_h_res_dims.data(), static_cast<std::uint8_t>(grad_h_res_dims.size()),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()), h_pre.data_ptr(),
                h_pre_dims.data(), static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_ch_dims.data(),
                static_cast<std::uint8_t>(r_ch_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_weight.data_ptr(), grad_weight_dims.data(),
                static_cast<std::uint8_t>(grad_weight_dims.size()), grad_alpha.data_ptr(), grad_alpha_dims.data(),
                static_cast<std::uint8_t>(grad_alpha_dims.size()), grad_bias.data_ptr(), grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), n, eps, compute_h_eps,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((grad_h_pre).scalar_type() == at::kFloat && (grad_h_post).scalar_type() == at::kFloat &&
            (grad_h_res).scalar_type() == at::kFloat && (grad_r).scalar_type() == at::kFloat &&
            (h_pre).scalar_type() == at::kFloat && (h_post).scalar_type() == at::kFloat &&
            (proj).scalar_type() == at::kFloat && (r).scalar_type() == at::kFloat && (x).scalar_type() == at::kFloat &&
            (weight).scalar_type() == at::kFloat && (alpha_pre).scalar_type() == at::kFloat &&
            (alpha_post).scalar_type() == at::kFloat && (alpha_res).scalar_type() == at::kFloat &&
            (grad_x).scalar_type() == at::kFloat && (grad_weight).scalar_type() == at::kFloat &&
            (grad_alpha).scalar_type() == at::kFloat && (grad_bias).scalar_type() == at::kFloat) {
            return callFp32ProjRmsComputeHBwd(
                grad_h_pre.data_ptr(), grad_h_pre_dims.data(), static_cast<std::uint8_t>(grad_h_pre_dims.size()),
                grad_h_post.data_ptr(), grad_h_post_dims.data(), static_cast<std::uint8_t>(grad_h_post_dims.size()),
                grad_h_res.data_ptr(), grad_h_res_dims.data(), static_cast<std::uint8_t>(grad_h_res_dims.size()),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()), h_pre.data_ptr(),
                h_pre_dims.data(), static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_ch_dims.data(),
                static_cast<std::uint8_t>(r_ch_dims.size()), x.data_ptr(), x_dims.data(),
                static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(), weight_dims.data(),
                static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), grad_weight.data_ptr(), grad_weight_dims.data(),
                static_cast<std::uint8_t>(grad_weight_dims.size()), grad_alpha.data_ptr(), grad_alpha_dims.data(),
                static_cast<std::uint8_t>(grad_alpha_dims.size()), grad_bias.data_ptr(), grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), n, eps, compute_h_eps,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (grad_alpha).scalar_type() == at::kFloat &&
            (grad_bias).scalar_type() == at::kFloat) {
            return callMixedFp32ProjRmsComputeHBwd(
                grad_h_pre.data_ptr(), grad_h_pre_dims.data(), static_cast<std::uint8_t>(grad_h_pre_dims.size()),
                numericTypeCode(grad_h_pre), grad_h_post.data_ptr(), grad_h_post_dims.data(),
                static_cast<std::uint8_t>(grad_h_post_dims.size()), numericTypeCode(grad_h_post), grad_h_res.data_ptr(),
                grad_h_res_dims.data(), static_cast<std::uint8_t>(grad_h_res_dims.size()), numericTypeCode(grad_h_res),
                grad_r.data_ptr(), grad_r_dims.data(), static_cast<std::uint8_t>(grad_r_dims.size()),
                numericTypeCode(grad_r), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), numericTypeCode(h_pre), h_post.data_ptr(),
                h_post_dims.data(), static_cast<std::uint8_t>(h_post_dims.size()), numericTypeCode(h_post),
                proj.data_ptr(), proj_dims.data(), static_cast<std::uint8_t>(proj_dims.size()), numericTypeCode(proj),
                r.data_ptr(), r_ch_dims.data(), static_cast<std::uint8_t>(r_ch_dims.size()), numericTypeCode(r),
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), numericTypeCode(x),
                weight.data_ptr(), weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()),
                numericTypeCode(weight), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), grad_x.data_ptr(), grad_x_dims.data(),
                static_cast<std::uint8_t>(grad_x_dims.size()), numericTypeCode(grad_x), grad_weight.data_ptr(),
                grad_weight_dims.data(), static_cast<std::uint8_t>(grad_weight_dims.size()),
                numericTypeCode(grad_weight), grad_alpha.data_ptr(), grad_alpha_dims.data(),
                static_cast<std::uint8_t>(grad_alpha_dims.size()), grad_bias.data_ptr(), grad_bias_dims.data(),
                static_cast<std::uint8_t>(grad_bias_dims.size()), n, eps, compute_h_eps,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.proj_rms_compute_h_bwd failed with status ",
                static_cast<unsigned int>(status));
    return {grad_x, grad_weight, grad_alpha, grad_bias};
}

std::vector<at::Tensor> projRmsComputeHFwd(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& alpha_pre,
                                           const at::Tensor& alpha_post, const at::Tensor& alpha_res,
                                           const at::Tensor& bias, std::int64_t n, double eps, double compute_h_eps) {
    checkDeviceTensor(x, "x");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(x.device().type()).getDevice() == x.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(x.device()));
    checkDeviceTensor(weight, "weight");
    checkDeviceTensor(alpha_pre, "alpha_pre");
    checkDeviceTensor(alpha_post, "alpha_post");
    checkDeviceTensor(alpha_res, "alpha_res");
    checkDeviceTensor(bias, "bias");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "proj_rms_compute_h_fwd expects x (M,K) and weight (N,K)");
    const std::int64_t rows    = x.size(0);
    const std::int64_t outputs = weight.size(0);
    at::Tensor h_pre           = torch::empty({rows, n}, x.options());
    at::Tensor h_post          = torch::empty({rows, n}, x.options());
    at::Tensor h_res           = torch::empty({rows, outputs - 2 * n}, x.options());
    at::Tensor r               = torch::empty({rows, 1}, x.options());
    at::Tensor proj            = torch::empty({rows, outputs}, x.options());
    const auto x_dims          = x.sizes().vec();
    TORCH_CHECK(weight.device() == x.device(), "all operands must be on the same device");
    const auto weight_dims = weight.sizes().vec();
    TORCH_CHECK(alpha_pre.device() == x.device(), "all operands must be on the same device");
    const auto alpha_pre_dims = alpha_pre.sizes().vec();
    TORCH_CHECK(alpha_post.device() == x.device(), "all operands must be on the same device");
    const auto alpha_post_dims = alpha_post.sizes().vec();
    TORCH_CHECK(alpha_res.device() == x.device(), "all operands must be on the same device");
    const auto alpha_res_dims = alpha_res.sizes().vec();
    TORCH_CHECK(bias.device() == x.device(), "all operands must be on the same device");
    const auto bias_dims = bias.sizes().vec();
    TORCH_CHECK(h_pre.device() == x.device(), "all operands must be on the same device");
    const auto h_pre_dims = h_pre.sizes().vec();
    TORCH_CHECK(h_post.device() == x.device(), "all operands must be on the same device");
    const auto h_post_dims = h_post.sizes().vec();
    TORCH_CHECK(h_res.device() == x.device(), "all operands must be on the same device");
    const auto h_res_dims = h_res.sizes().vec();
    TORCH_CHECK(r.device() == x.device(), "all operands must be on the same device");
    const auto r_dims = r.sizes().vec();
    TORCH_CHECK(proj.device() == x.device(), "all operands must be on the same device");
    const auto proj_dims             = proj.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((x).scalar_type() == at::kBFloat16 && (weight).scalar_type() == at::kBFloat16 &&
            (alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (bias).scalar_type() == at::kFloat &&
            (h_pre).scalar_type() == at::kBFloat16 && (h_post).scalar_type() == at::kBFloat16 &&
            (h_res).scalar_type() == at::kBFloat16 && (r).scalar_type() == at::kBFloat16 &&
            (proj).scalar_type() == at::kBFloat16) {
            return queryBf16ProjRmsComputeHFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(),
                weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(),
                alpha_pre_dims.data(), static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(),
                alpha_post_dims.data(), static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(),
                alpha_res_dims.data(), static_cast<std::uint8_t>(alpha_res_dims.size()), bias.data_ptr(),
                bias_dims.data(), static_cast<std::uint8_t>(bias_dims.size()), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), h_res.data_ptr(), h_res_dims.data(),
                static_cast<std::uint8_t>(h_res_dims.size()), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), n, eps, compute_h_eps, &required);
        }
        if ((x).scalar_type() == at::kHalf && (weight).scalar_type() == at::kHalf &&
            (alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (bias).scalar_type() == at::kFloat &&
            (h_pre).scalar_type() == at::kHalf && (h_post).scalar_type() == at::kHalf &&
            (h_res).scalar_type() == at::kHalf && (r).scalar_type() == at::kHalf && (proj).scalar_type() == at::kHalf) {
            return queryFp16ProjRmsComputeHFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(),
                weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(),
                alpha_pre_dims.data(), static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(),
                alpha_post_dims.data(), static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(),
                alpha_res_dims.data(), static_cast<std::uint8_t>(alpha_res_dims.size()), bias.data_ptr(),
                bias_dims.data(), static_cast<std::uint8_t>(bias_dims.size()), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), h_res.data_ptr(), h_res_dims.data(),
                static_cast<std::uint8_t>(h_res_dims.size()), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), n, eps, compute_h_eps, &required);
        }
        if ((x).scalar_type() == at::kFloat && (weight).scalar_type() == at::kFloat &&
            (alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (bias).scalar_type() == at::kFloat &&
            (h_pre).scalar_type() == at::kFloat && (h_post).scalar_type() == at::kFloat &&
            (h_res).scalar_type() == at::kFloat && (r).scalar_type() == at::kFloat &&
            (proj).scalar_type() == at::kFloat) {
            return queryFp32ProjRmsComputeHFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(),
                weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(),
                alpha_pre_dims.data(), static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(),
                alpha_post_dims.data(), static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(),
                alpha_res_dims.data(), static_cast<std::uint8_t>(alpha_res_dims.size()), bias.data_ptr(),
                bias_dims.data(), static_cast<std::uint8_t>(bias_dims.size()), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), h_res.data_ptr(), h_res_dims.data(),
                static_cast<std::uint8_t>(h_res_dims.size()), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), n, eps, compute_h_eps, &required);
        }
        if ((alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (bias).scalar_type() == at::kFloat) {
            return queryMixedFp32ProjRmsComputeHFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), numericTypeCode(x),
                weight.data_ptr(), weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()),
                numericTypeCode(weight), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), bias.data_ptr(), bias_dims.data(),
                static_cast<std::uint8_t>(bias_dims.size()), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), numericTypeCode(h_pre), h_post.data_ptr(),
                h_post_dims.data(), static_cast<std::uint8_t>(h_post_dims.size()), numericTypeCode(h_post),
                h_res.data_ptr(), h_res_dims.data(), static_cast<std::uint8_t>(h_res_dims.size()),
                numericTypeCode(h_res), r.data_ptr(), r_dims.data(), static_cast<std::uint8_t>(r_dims.size()),
                numericTypeCode(r), proj.data_ptr(), proj_dims.data(), static_cast<std::uint8_t>(proj_dims.size()),
                numericTypeCode(proj), n, eps, compute_h_eps, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, x.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((x).scalar_type() == at::kBFloat16 && (weight).scalar_type() == at::kBFloat16 &&
            (alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (bias).scalar_type() == at::kFloat &&
            (h_pre).scalar_type() == at::kBFloat16 && (h_post).scalar_type() == at::kBFloat16 &&
            (h_res).scalar_type() == at::kBFloat16 && (r).scalar_type() == at::kBFloat16 &&
            (proj).scalar_type() == at::kBFloat16) {
            return callBf16ProjRmsComputeHFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(),
                weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(),
                alpha_pre_dims.data(), static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(),
                alpha_post_dims.data(), static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(),
                alpha_res_dims.data(), static_cast<std::uint8_t>(alpha_res_dims.size()), bias.data_ptr(),
                bias_dims.data(), static_cast<std::uint8_t>(bias_dims.size()), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), h_res.data_ptr(), h_res_dims.data(),
                static_cast<std::uint8_t>(h_res_dims.size()), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), n, eps, compute_h_eps,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((x).scalar_type() == at::kHalf && (weight).scalar_type() == at::kHalf &&
            (alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (bias).scalar_type() == at::kFloat &&
            (h_pre).scalar_type() == at::kHalf && (h_post).scalar_type() == at::kHalf &&
            (h_res).scalar_type() == at::kHalf && (r).scalar_type() == at::kHalf && (proj).scalar_type() == at::kHalf) {
            return callFp16ProjRmsComputeHFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(),
                weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(),
                alpha_pre_dims.data(), static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(),
                alpha_post_dims.data(), static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(),
                alpha_res_dims.data(), static_cast<std::uint8_t>(alpha_res_dims.size()), bias.data_ptr(),
                bias_dims.data(), static_cast<std::uint8_t>(bias_dims.size()), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), h_res.data_ptr(), h_res_dims.data(),
                static_cast<std::uint8_t>(h_res_dims.size()), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), n, eps, compute_h_eps,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((x).scalar_type() == at::kFloat && (weight).scalar_type() == at::kFloat &&
            (alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (bias).scalar_type() == at::kFloat &&
            (h_pre).scalar_type() == at::kFloat && (h_post).scalar_type() == at::kFloat &&
            (h_res).scalar_type() == at::kFloat && (r).scalar_type() == at::kFloat &&
            (proj).scalar_type() == at::kFloat) {
            return callFp32ProjRmsComputeHFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), weight.data_ptr(),
                weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()), alpha_pre.data_ptr(),
                alpha_pre_dims.data(), static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(),
                alpha_post_dims.data(), static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(),
                alpha_res_dims.data(), static_cast<std::uint8_t>(alpha_res_dims.size()), bias.data_ptr(),
                bias_dims.data(), static_cast<std::uint8_t>(bias_dims.size()), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), h_post.data_ptr(), h_post_dims.data(),
                static_cast<std::uint8_t>(h_post_dims.size()), h_res.data_ptr(), h_res_dims.data(),
                static_cast<std::uint8_t>(h_res_dims.size()), r.data_ptr(), r_dims.data(),
                static_cast<std::uint8_t>(r_dims.size()), proj.data_ptr(), proj_dims.data(),
                static_cast<std::uint8_t>(proj_dims.size()), n, eps, compute_h_eps,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((alpha_pre).scalar_type() == at::kFloat && (alpha_post).scalar_type() == at::kFloat &&
            (alpha_res).scalar_type() == at::kFloat && (bias).scalar_type() == at::kFloat) {
            return callMixedFp32ProjRmsComputeHFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), numericTypeCode(x),
                weight.data_ptr(), weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()),
                numericTypeCode(weight), alpha_pre.data_ptr(), alpha_pre_dims.data(),
                static_cast<std::uint8_t>(alpha_pre_dims.size()), alpha_post.data_ptr(), alpha_post_dims.data(),
                static_cast<std::uint8_t>(alpha_post_dims.size()), alpha_res.data_ptr(), alpha_res_dims.data(),
                static_cast<std::uint8_t>(alpha_res_dims.size()), bias.data_ptr(), bias_dims.data(),
                static_cast<std::uint8_t>(bias_dims.size()), h_pre.data_ptr(), h_pre_dims.data(),
                static_cast<std::uint8_t>(h_pre_dims.size()), numericTypeCode(h_pre), h_post.data_ptr(),
                h_post_dims.data(), static_cast<std::uint8_t>(h_post_dims.size()), numericTypeCode(h_post),
                h_res.data_ptr(), h_res_dims.data(), static_cast<std::uint8_t>(h_res_dims.size()),
                numericTypeCode(h_res), r.data_ptr(), r_dims.data(), static_cast<std::uint8_t>(r_dims.size()),
                numericTypeCode(r), proj.data_ptr(), proj_dims.data(), static_cast<std::uint8_t>(proj_dims.size()),
                numericTypeCode(proj), n, eps, compute_h_eps, (required == 0 ? nullptr : workspace.data_ptr()),
                required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.proj_rms_compute_h_fwd failed with status ",
                static_cast<unsigned int>(status));
    return {h_pre, h_post, h_res, r, proj};
}

std::vector<at::Tensor> projRmsFwd(const at::Tensor& x, const at::Tensor& weight, double eps) {
    checkDeviceTensor(x, "x");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(x.device().type()).getDevice() == x.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(x.device()));
    checkDeviceTensor(weight, "weight");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "proj_rms_fwd expects x (M,K) and weight (N,K)");
    at::Tensor proj   = torch::empty({x.size(0), weight.size(0)}, x.options());
    at::Tensor r      = torch::empty({x.size(0), 1}, x.options());
    const auto x_dims = x.sizes().vec();
    TORCH_CHECK(weight.device() == x.device(), "all operands must be on the same device");
    const auto weight_dims = weight.sizes().vec();
    TORCH_CHECK(proj.device() == x.device(), "all operands must be on the same device");
    const auto proj_dims = proj.sizes().vec();
    TORCH_CHECK(r.device() == x.device(), "all operands must be on the same device");
    const auto r_dims                = r.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((x).scalar_type() == at::kBFloat16 && (weight).scalar_type() == at::kBFloat16 &&
            (proj).scalar_type() == at::kBFloat16 && (r).scalar_type() == at::kBFloat16) {
            return queryBf16ProjRmsFwd(x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()),
                                       weight.data_ptr(), weight_dims.data(),
                                       static_cast<std::uint8_t>(weight_dims.size()), proj.data_ptr(), proj_dims.data(),
                                       static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_dims.data(),
                                       static_cast<std::uint8_t>(r_dims.size()), eps, &required);
        }
        if ((x).scalar_type() == at::kHalf && (weight).scalar_type() == at::kHalf &&
            (proj).scalar_type() == at::kHalf && (r).scalar_type() == at::kHalf) {
            return queryFp16ProjRmsFwd(x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()),
                                       weight.data_ptr(), weight_dims.data(),
                                       static_cast<std::uint8_t>(weight_dims.size()), proj.data_ptr(), proj_dims.data(),
                                       static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_dims.data(),
                                       static_cast<std::uint8_t>(r_dims.size()), eps, &required);
        }
        if ((x).scalar_type() == at::kFloat && (weight).scalar_type() == at::kFloat &&
            (proj).scalar_type() == at::kFloat && (r).scalar_type() == at::kFloat) {
            return queryFp32ProjRmsFwd(x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()),
                                       weight.data_ptr(), weight_dims.data(),
                                       static_cast<std::uint8_t>(weight_dims.size()), proj.data_ptr(), proj_dims.data(),
                                       static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_dims.data(),
                                       static_cast<std::uint8_t>(r_dims.size()), eps, &required);
        }
        {
            return queryMixedFp32ProjRmsFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), numericTypeCode(x),
                weight.data_ptr(), weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()),
                numericTypeCode(weight), proj.data_ptr(), proj_dims.data(), static_cast<std::uint8_t>(proj_dims.size()),
                numericTypeCode(proj), r.data_ptr(), r_dims.data(), static_cast<std::uint8_t>(r_dims.size()),
                numericTypeCode(r), eps, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, x.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((x).scalar_type() == at::kBFloat16 && (weight).scalar_type() == at::kBFloat16 &&
            (proj).scalar_type() == at::kBFloat16 && (r).scalar_type() == at::kBFloat16) {
            return callBf16ProjRmsFwd(x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()),
                                      weight.data_ptr(), weight_dims.data(),
                                      static_cast<std::uint8_t>(weight_dims.size()), proj.data_ptr(), proj_dims.data(),
                                      static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_dims.data(),
                                      static_cast<std::uint8_t>(r_dims.size()), eps,
                                      (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((x).scalar_type() == at::kHalf && (weight).scalar_type() == at::kHalf &&
            (proj).scalar_type() == at::kHalf && (r).scalar_type() == at::kHalf) {
            return callFp16ProjRmsFwd(x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()),
                                      weight.data_ptr(), weight_dims.data(),
                                      static_cast<std::uint8_t>(weight_dims.size()), proj.data_ptr(), proj_dims.data(),
                                      static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_dims.data(),
                                      static_cast<std::uint8_t>(r_dims.size()), eps,
                                      (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((x).scalar_type() == at::kFloat && (weight).scalar_type() == at::kFloat &&
            (proj).scalar_type() == at::kFloat && (r).scalar_type() == at::kFloat) {
            return callFp32ProjRmsFwd(x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()),
                                      weight.data_ptr(), weight_dims.data(),
                                      static_cast<std::uint8_t>(weight_dims.size()), proj.data_ptr(), proj_dims.data(),
                                      static_cast<std::uint8_t>(proj_dims.size()), r.data_ptr(), r_dims.data(),
                                      static_cast<std::uint8_t>(r_dims.size()), eps,
                                      (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        {
            return callMixedFp32ProjRmsFwd(
                x.data_ptr(), x_dims.data(), static_cast<std::uint8_t>(x_dims.size()), numericTypeCode(x),
                weight.data_ptr(), weight_dims.data(), static_cast<std::uint8_t>(weight_dims.size()),
                numericTypeCode(weight), proj.data_ptr(), proj_dims.data(), static_cast<std::uint8_t>(proj_dims.size()),
                numericTypeCode(proj), r.data_ptr(), r_dims.data(), static_cast<std::uint8_t>(r_dims.size()),
                numericTypeCode(r), eps, (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.proj_rms_fwd failed with status ", static_cast<unsigned int>(status));
    return {proj, r};
}

at::Tensor sinkhornBwd(const at::Tensor& grad_output, const at::Tensor& m_init, std::int64_t num_iterations,
                       double eps) {
    checkDeviceTensor(grad_output, "grad_output");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(grad_output.device().type()).getDevice() ==
                                                        grad_output.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(grad_output.device()));
    checkDeviceTensor(m_init, "m_init");
    at::Tensor grad_logits      = torch::empty_like(grad_output);
    const auto grad_output_dims = grad_output.sizes().vec();
    TORCH_CHECK(m_init.device() == grad_output.device(), "all operands must be on the same device");
    const auto m_init_dims = m_init.sizes().vec();
    TORCH_CHECK(grad_logits.device() == grad_output.device(), "all operands must be on the same device");
    const auto grad_logits_dims = grad_logits.sizes().vec();
    const unsigned char status  = [&]() -> unsigned char {
        if ((grad_output).scalar_type() == at::kBFloat16 && (m_init).scalar_type() == at::kFloat &&
            (grad_logits).scalar_type() == at::kBFloat16) {
            return callBf16SinkhornBwd(
                grad_output.data_ptr(), grad_output_dims.data(), static_cast<std::uint8_t>(grad_output_dims.size()),
                m_init.data_ptr(), m_init_dims.data(), static_cast<std::uint8_t>(m_init_dims.size()),
                grad_logits.data_ptr(), grad_logits_dims.data(), static_cast<std::uint8_t>(grad_logits_dims.size()),
                num_iterations, eps, currentStream());
        }
        if ((grad_output).scalar_type() == at::kHalf && (m_init).scalar_type() == at::kFloat &&
            (grad_logits).scalar_type() == at::kHalf) {
            return callFp16SinkhornBwd(
                grad_output.data_ptr(), grad_output_dims.data(), static_cast<std::uint8_t>(grad_output_dims.size()),
                m_init.data_ptr(), m_init_dims.data(), static_cast<std::uint8_t>(m_init_dims.size()),
                grad_logits.data_ptr(), grad_logits_dims.data(), static_cast<std::uint8_t>(grad_logits_dims.size()),
                num_iterations, eps, currentStream());
        }
        if ((grad_output).scalar_type() == at::kFloat && (m_init).scalar_type() == at::kFloat &&
            (grad_logits).scalar_type() == at::kFloat) {
            return callFp32SinkhornBwd(
                grad_output.data_ptr(), grad_output_dims.data(), static_cast<std::uint8_t>(grad_output_dims.size()),
                m_init.data_ptr(), m_init_dims.data(), static_cast<std::uint8_t>(m_init_dims.size()),
                grad_logits.data_ptr(), grad_logits_dims.data(), static_cast<std::uint8_t>(grad_logits_dims.size()),
                num_iterations, eps, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.sinkhorn_bwd failed with status ", static_cast<unsigned int>(status));
    return grad_logits;
}

std::vector<at::Tensor> sinkhornFwd(const at::Tensor& logits, std::int64_t num_iterations, double eps) {
    checkDeviceTensor(logits, "logits");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(logits.device().type()).getDevice() ==
                                                        logits.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(logits.device()));
    TORCH_CHECK(logits.dim() >= 2 && logits.size(-1) > 0, "sinkhorn_fwd expects matrices with a positive width");
    const std::int64_t size  = logits.size(-1);
    const std::int64_t batch = logits.numel() / size / size;
    at::Tensor output        = torch::empty_like(logits);
    at::Tensor m_init        = torch::empty({batch, size, size}, logits.options().dtype(at::kFloat));
    const auto logits_dims   = logits.sizes().vec();
    TORCH_CHECK(output.device() == logits.device(), "all operands must be on the same device");
    const auto output_dims = output.sizes().vec();
    TORCH_CHECK(m_init.device() == logits.device(), "all operands must be on the same device");
    const auto m_init_dims     = m_init.sizes().vec();
    const unsigned char status = [&]() -> unsigned char {
        if ((logits).scalar_type() == at::kBFloat16 && (output).scalar_type() == at::kBFloat16 &&
            (m_init).scalar_type() == at::kFloat) {
            return callBf16SinkhornFwd(
                logits.data_ptr(), logits_dims.data(), static_cast<std::uint8_t>(logits_dims.size()), output.data_ptr(),
                output_dims.data(), static_cast<std::uint8_t>(output_dims.size()), m_init.data_ptr(),
                m_init_dims.data(), static_cast<std::uint8_t>(m_init_dims.size()), num_iterations, eps,
                currentStream());
        }
        if ((logits).scalar_type() == at::kHalf && (output).scalar_type() == at::kHalf &&
            (m_init).scalar_type() == at::kFloat) {
            return callFp16SinkhornFwd(
                logits.data_ptr(), logits_dims.data(), static_cast<std::uint8_t>(logits_dims.size()), output.data_ptr(),
                output_dims.data(), static_cast<std::uint8_t>(output_dims.size()), m_init.data_ptr(),
                m_init_dims.data(), static_cast<std::uint8_t>(m_init_dims.size()), num_iterations, eps,
                currentStream());
        }
        if ((logits).scalar_type() == at::kFloat && (output).scalar_type() == at::kFloat &&
            (m_init).scalar_type() == at::kFloat) {
            return callFp32SinkhornFwd(
                logits.data_ptr(), logits_dims.data(), static_cast<std::uint8_t>(logits_dims.size()), output.data_ptr(),
                output_dims.data(), static_cast<std::uint8_t>(output_dims.size()), m_init.data_ptr(),
                m_init_dims.data(), static_cast<std::uint8_t>(m_init_dims.size()), num_iterations, eps,
                currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.sinkhorn_fwd failed with status ", static_cast<unsigned int>(status));
    return {output, m_init};
}

std::vector<at::Tensor> sparseAttnBwd(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& out,
                                      const at::Tensor& dout, const at::Tensor& lse, const at::Tensor& attn_sink,
                                      const at::Tensor& topk_indices, double softmax_scale,
                                      const std::optional<at::Tensor>& topk_length) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(kv, "kv");
    checkDeviceTensor(out, "out");
    checkDeviceTensor(dout, "dout");
    checkDeviceTensor(lse, "lse");
    checkDeviceTensor(attn_sink, "attn_sink");
    checkDeviceTensor(topk_indices, "topk_indices");
    TORCH_CHECK(q.dim() == 3 && (kv.dim() == 2 || kv.dim() == 3) && topk_indices.dim() == 2,
                "sparse_attn_bwd expects the sparse_attn_fwd ports plus dout");
    if (topk_length.has_value()) { checkDeviceTensor(*topk_length, "topk_length"); }

    at::Tensor dq                = torch::empty_like(q);
    at::Tensor dkv               = torch::empty_like(kv);
    at::Tensor dsink             = torch::empty_like(attn_sink);
    const auto q_dims            = q.sizes().vec();
    const auto kv_dims           = kv.sizes().vec();
    const auto out_dims          = out.sizes().vec();
    const auto dout_dims         = dout.sizes().vec();
    const auto lse_dims          = lse.sizes().vec();
    const auto attn_sink_dims    = attn_sink.sizes().vec();
    const auto topk_indices_dims = topk_indices.sizes().vec();
    const auto topk_length_dims =
        (topk_length.has_value()) ? (*topk_length).sizes().vec() : std::vector<std::int64_t>{q.size(0)};
    const void* topk_length_data = (topk_length.has_value()) ? (*topk_length).data_ptr() : nullptr;
    const auto topk_length_type  = (topk_length.has_value()) ? (*topk_length).scalar_type() : at::kInt;
    const auto dq_dims           = dq.sizes().vec();
    const auto dkv_dims          = dkv.sizes().vec();
    const auto dsink_dims        = dsink.sizes().vec();

    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (kv).scalar_type() == at::kBFloat16 &&
            (out).scalar_type() == at::kBFloat16 && (dout).scalar_type() == at::kBFloat16 &&
            (lse).scalar_type() == at::kFloat && (attn_sink).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && topk_length_type == at::kInt &&
            (dq).scalar_type() == at::kBFloat16 && (dkv).scalar_type() == at::kBFloat16 &&
            (dsink).scalar_type() == at::kFloat) {
            return queryBf16SparseAttnBwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), dout.data_ptr(), dout_dims.data(),
                static_cast<std::uint8_t>(dout_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dkv.data_ptr(), dkv_dims.data(),
                static_cast<std::uint8_t>(dkv_dims.size()), dsink.data_ptr(), dsink_dims.data(),
                static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale, &required);
        }
        if ((q).scalar_type() == at::kHalf && (kv).scalar_type() == at::kHalf && (out).scalar_type() == at::kHalf &&
            (dout).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (dq).scalar_type() == at::kHalf && (dkv).scalar_type() == at::kHalf &&
            (dsink).scalar_type() == at::kFloat) {
            return queryFp16SparseAttnBwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), dout.data_ptr(), dout_dims.data(),
                static_cast<std::uint8_t>(dout_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dkv.data_ptr(), dkv_dims.data(),
                static_cast<std::uint8_t>(dkv_dims.size()), dsink.data_ptr(), dsink_dims.data(),
                static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale, &required);
        }
        if ((q).scalar_type() == at::kFloat && (kv).scalar_type() == at::kFloat && (out).scalar_type() == at::kFloat &&
            (dout).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (dq).scalar_type() == at::kFloat && (dkv).scalar_type() == at::kFloat &&
            (dsink).scalar_type() == at::kFloat) {
            return queryFp32SparseAttnBwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), dout.data_ptr(), dout_dims.data(),
                static_cast<std::uint8_t>(dout_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dkv.data_ptr(), dkv_dims.data(),
                static_cast<std::uint8_t>(dkv_dims.size()), dsink.data_ptr(), dsink_dims.data(),
                static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale, &required);
        }
        if ((lse).scalar_type() == at::kFloat && (attn_sink).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && topk_length_type == at::kInt &&
            (dsink).scalar_type() == at::kFloat) {
            return queryMixedFp32SparseAttnBwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q),
                kv.data_ptr(), kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), numericTypeCode(kv),
                out.data_ptr(), out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), numericTypeCode(out),
                dout.data_ptr(), dout_dims.data(), static_cast<std::uint8_t>(dout_dims.size()), numericTypeCode(dout),
                lse.data_ptr(), lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(),
                attn_sink_dims.data(), static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(),
                topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data,
                topk_length_dims.data(), static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dkv.data_ptr(),
                dkv_dims.data(), static_cast<std::uint8_t>(dkv_dims.size()), numericTypeCode(dkv), dsink.data_ptr(),
                dsink_dims.data(), static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, dq.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (kv).scalar_type() == at::kBFloat16 &&
            (out).scalar_type() == at::kBFloat16 && (dout).scalar_type() == at::kBFloat16 &&
            (lse).scalar_type() == at::kFloat && (attn_sink).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && topk_length_type == at::kInt &&
            (dq).scalar_type() == at::kBFloat16 && (dkv).scalar_type() == at::kBFloat16 &&
            (dsink).scalar_type() == at::kFloat) {
            return callBf16SparseAttnBwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), dout.data_ptr(), dout_dims.data(),
                static_cast<std::uint8_t>(dout_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dkv.data_ptr(), dkv_dims.data(),
                static_cast<std::uint8_t>(dkv_dims.size()), dsink.data_ptr(), dsink_dims.data(),
                static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (kv).scalar_type() == at::kHalf && (out).scalar_type() == at::kHalf &&
            (dout).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (dq).scalar_type() == at::kHalf && (dkv).scalar_type() == at::kHalf &&
            (dsink).scalar_type() == at::kFloat) {
            return callFp16SparseAttnBwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), dout.data_ptr(), dout_dims.data(),
                static_cast<std::uint8_t>(dout_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dkv.data_ptr(), dkv_dims.data(),
                static_cast<std::uint8_t>(dkv_dims.size()), dsink.data_ptr(), dsink_dims.data(),
                static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (kv).scalar_type() == at::kFloat && (out).scalar_type() == at::kFloat &&
            (dout).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (dq).scalar_type() == at::kFloat && (dkv).scalar_type() == at::kFloat &&
            (dsink).scalar_type() == at::kFloat) {
            return callFp32SparseAttnBwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), dout.data_ptr(), dout_dims.data(),
                static_cast<std::uint8_t>(dout_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(), dq_dims.data(),
                static_cast<std::uint8_t>(dq_dims.size()), dkv.data_ptr(), dkv_dims.data(),
                static_cast<std::uint8_t>(dkv_dims.size()), dsink.data_ptr(), dsink_dims.data(),
                static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((lse).scalar_type() == at::kFloat && (attn_sink).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && topk_length_type == at::kInt &&
            (dsink).scalar_type() == at::kFloat) {
            return callMixedFp32SparseAttnBwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q),
                kv.data_ptr(), kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), numericTypeCode(kv),
                out.data_ptr(), out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), numericTypeCode(out),
                dout.data_ptr(), dout_dims.data(), static_cast<std::uint8_t>(dout_dims.size()), numericTypeCode(dout),
                lse.data_ptr(), lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(),
                attn_sink_dims.data(), static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(),
                topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data,
                topk_length_dims.data(), static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(),
                dq_dims.data(), static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dkv.data_ptr(),
                dkv_dims.data(), static_cast<std::uint8_t>(dkv_dims.size()), numericTypeCode(dkv), dsink.data_ptr(),
                dsink_dims.data(), static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.sparse_attn_bwd failed with status ",
                static_cast<unsigned int>(status));
    return {dq, dkv, dsink};
}

std::vector<at::Tensor> sparseAttnFwd(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& attn_sink,
                                      const at::Tensor& topk_indices, double softmax_scale,
                                      const std::optional<at::Tensor>& topk_length) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(kv, "kv");
    checkDeviceTensor(attn_sink, "attn_sink");
    checkDeviceTensor(topk_indices, "topk_indices");
    TORCH_CHECK(q.dim() == 3 && (kv.dim() == 2 || kv.dim() == 3) && topk_indices.dim() == 2,
                "sparse_attn_fwd expects q (T,H,D), kv (T_kv,D) or (T_kv,1,D), topk_idxs (T,topk)");
    if (topk_length.has_value()) { checkDeviceTensor(*topk_length, "topk_length"); }

    at::Tensor out               = torch::empty({q.size(0), q.size(1), 512}, q.options());
    at::Tensor lse               = torch::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    const auto q_dims            = q.sizes().vec();
    const auto kv_dims           = kv.sizes().vec();
    const auto attn_sink_dims    = attn_sink.sizes().vec();
    const auto topk_indices_dims = topk_indices.sizes().vec();
    const auto topk_length_dims =
        (topk_length.has_value()) ? (*topk_length).sizes().vec() : std::vector<std::int64_t>{q.size(0)};
    const void* topk_length_data     = (topk_length.has_value()) ? (*topk_length).data_ptr() : nullptr;
    const auto topk_length_type      = (topk_length.has_value()) ? (*topk_length).scalar_type() : at::kInt;
    const auto out_dims              = out.sizes().vec();
    const auto lse_dims              = lse.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (kv).scalar_type() == at::kBFloat16 &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kBFloat16 && (lse).scalar_type() == at::kFloat) {
            return queryBf16SparseAttnFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), softmax_scale, &required);
        }
        if ((q).scalar_type() == at::kHalf && (kv).scalar_type() == at::kHalf &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat) {
            return queryFp16SparseAttnFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), softmax_scale, &required);
        }
        if ((q).scalar_type() == at::kFloat && (kv).scalar_type() == at::kFloat &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat) {
            return queryFp32SparseAttnFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), softmax_scale, &required);
        }
        if ((attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (lse).scalar_type() == at::kFloat) {
            return queryMixedFp32SparseAttnFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q),
                kv.data_ptr(), kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), numericTypeCode(kv),
                attn_sink.data_ptr(), attn_sink_dims.data(), static_cast<std::uint8_t>(attn_sink_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                topk_length_data, topk_length_dims.data(), static_cast<std::uint8_t>(topk_length_dims.size()),
                out.data_ptr(), out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), numericTypeCode(out),
                lse.data_ptr(), lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), softmax_scale, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, q.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (kv).scalar_type() == at::kBFloat16 &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kBFloat16 && (lse).scalar_type() == at::kFloat) {
            return callBf16SparseAttnFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), softmax_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (kv).scalar_type() == at::kHalf &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat) {
            return callFp16SparseAttnFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), softmax_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (kv).scalar_type() == at::kFloat &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat) {
            return callFp32SparseAttnFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), softmax_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (lse).scalar_type() == at::kFloat) {
            return callMixedFp32SparseAttnFwd(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q),
                kv.data_ptr(), kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), numericTypeCode(kv),
                attn_sink.data_ptr(), attn_sink_dims.data(), static_cast<std::uint8_t>(attn_sink_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                topk_length_data, topk_length_dims.data(), static_cast<std::uint8_t>(topk_length_dims.size()),
                out.data_ptr(), out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), numericTypeCode(out),
                lse.data_ptr(), lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), softmax_scale,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.sparse_attn_fwd failed with status ",
                static_cast<unsigned int>(status));
    return {out, lse};
}

std::vector<at::Tensor> sparseAttnFwdIndexer(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& attn_sink,
                                             const at::Tensor& topk_indices, double softmax_scale,
                                             std::int64_t indexer_topk, const std::optional<at::Tensor>& topk_length) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(kv, "kv");
    checkDeviceTensor(attn_sink, "attn_sink");
    checkDeviceTensor(topk_indices, "topk_indices");
    TORCH_CHECK(q.dim() == 3 && (kv.dim() == 2 || kv.dim() == 3) && topk_indices.dim() == 2 && indexer_topk > 0,
                "sparse_attn_fwd_indexer expects the sparse_attn_fwd ports and indexer_topk > 0");
    if (topk_length.has_value()) { checkDeviceTensor(*topk_length, "topk_length"); }

    at::Tensor out               = torch::empty({q.size(0), q.size(1), 512}, q.options());
    at::Tensor max_logits        = torch::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    at::Tensor lse               = torch::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    at::Tensor lse_indexer       = torch::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    const auto q_dims            = q.sizes().vec();
    const auto kv_dims           = kv.sizes().vec();
    const auto attn_sink_dims    = attn_sink.sizes().vec();
    const auto topk_indices_dims = topk_indices.sizes().vec();
    const auto topk_length_dims =
        (topk_length.has_value()) ? (*topk_length).sizes().vec() : std::vector<std::int64_t>{q.size(0)};
    const void* topk_length_data     = (topk_length.has_value()) ? (*topk_length).data_ptr() : nullptr;
    const auto topk_length_type      = (topk_length.has_value()) ? (*topk_length).scalar_type() : at::kInt;
    const auto out_dims              = out.sizes().vec();
    const auto max_logits_dims       = max_logits.sizes().vec();
    const auto lse_dims              = lse.sizes().vec();
    const auto lse_indexer_dims      = lse_indexer.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (kv).scalar_type() == at::kBFloat16 &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kBFloat16 &&
            (max_logits).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (lse_indexer).scalar_type() == at::kFloat) {
            return queryBf16SparseAttnFwdIndexer(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), max_logits.data_ptr(), max_logits_dims.data(),
                static_cast<std::uint8_t>(max_logits_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), lse_indexer.data_ptr(), lse_indexer_dims.data(),
                static_cast<std::uint8_t>(lse_indexer_dims.size()), softmax_scale, indexer_topk, &required);
        }
        if ((q).scalar_type() == at::kHalf && (kv).scalar_type() == at::kHalf &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kHalf &&
            (max_logits).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (lse_indexer).scalar_type() == at::kFloat) {
            return queryFp16SparseAttnFwdIndexer(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), max_logits.data_ptr(), max_logits_dims.data(),
                static_cast<std::uint8_t>(max_logits_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), lse_indexer.data_ptr(), lse_indexer_dims.data(),
                static_cast<std::uint8_t>(lse_indexer_dims.size()), softmax_scale, indexer_topk, &required);
        }
        if ((q).scalar_type() == at::kFloat && (kv).scalar_type() == at::kFloat &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (max_logits).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (lse_indexer).scalar_type() == at::kFloat) {
            return queryFp32SparseAttnFwdIndexer(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), max_logits.data_ptr(), max_logits_dims.data(),
                static_cast<std::uint8_t>(max_logits_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), lse_indexer.data_ptr(), lse_indexer_dims.data(),
                static_cast<std::uint8_t>(lse_indexer_dims.size()), softmax_scale, indexer_topk, &required);
        }
        if ((attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (max_logits).scalar_type() == at::kFloat &&
            (lse).scalar_type() == at::kFloat && (lse_indexer).scalar_type() == at::kFloat) {
            return queryMixedFp32SparseAttnFwdIndexer(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q),
                kv.data_ptr(), kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), numericTypeCode(kv),
                attn_sink.data_ptr(), attn_sink_dims.data(), static_cast<std::uint8_t>(attn_sink_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                topk_length_data, topk_length_dims.data(), static_cast<std::uint8_t>(topk_length_dims.size()),
                out.data_ptr(), out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), numericTypeCode(out),
                max_logits.data_ptr(), max_logits_dims.data(), static_cast<std::uint8_t>(max_logits_dims.size()),
                lse.data_ptr(), lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), lse_indexer.data_ptr(),
                lse_indexer_dims.data(), static_cast<std::uint8_t>(lse_indexer_dims.size()), softmax_scale,
                indexer_topk, &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, q.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (kv).scalar_type() == at::kBFloat16 &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kBFloat16 &&
            (max_logits).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (lse_indexer).scalar_type() == at::kFloat) {
            return callBf16SparseAttnFwdIndexer(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), max_logits.data_ptr(), max_logits_dims.data(),
                static_cast<std::uint8_t>(max_logits_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), lse_indexer.data_ptr(), lse_indexer_dims.data(),
                static_cast<std::uint8_t>(lse_indexer_dims.size()), softmax_scale, indexer_topk,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (kv).scalar_type() == at::kHalf &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kHalf &&
            (max_logits).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (lse_indexer).scalar_type() == at::kFloat) {
            return callFp16SparseAttnFwdIndexer(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), max_logits.data_ptr(), max_logits_dims.data(),
                static_cast<std::uint8_t>(max_logits_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), lse_indexer.data_ptr(), lse_indexer_dims.data(),
                static_cast<std::uint8_t>(lse_indexer_dims.size()), softmax_scale, indexer_topk,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (kv).scalar_type() == at::kFloat &&
            (attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (out).scalar_type() == at::kFloat &&
            (max_logits).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (lse_indexer).scalar_type() == at::kFloat) {
            return callFp32SparseAttnFwdIndexer(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), kv.data_ptr(), kv_dims.data(),
                static_cast<std::uint8_t>(kv_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
                static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
                static_cast<std::uint8_t>(topk_length_dims.size()), out.data_ptr(), out_dims.data(),
                static_cast<std::uint8_t>(out_dims.size()), max_logits.data_ptr(), max_logits_dims.data(),
                static_cast<std::uint8_t>(max_logits_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), lse_indexer.data_ptr(), lse_indexer_dims.data(),
                static_cast<std::uint8_t>(lse_indexer_dims.size()), softmax_scale, indexer_topk,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((attn_sink).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            topk_length_type == at::kInt && (max_logits).scalar_type() == at::kFloat &&
            (lse).scalar_type() == at::kFloat && (lse_indexer).scalar_type() == at::kFloat) {
            return callMixedFp32SparseAttnFwdIndexer(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q),
                kv.data_ptr(), kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), numericTypeCode(kv),
                attn_sink.data_ptr(), attn_sink_dims.data(), static_cast<std::uint8_t>(attn_sink_dims.size()),
                topk_indices.data_ptr(), topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()),
                topk_length_data, topk_length_dims.data(), static_cast<std::uint8_t>(topk_length_dims.size()),
                out.data_ptr(), out_dims.data(), static_cast<std::uint8_t>(out_dims.size()), numericTypeCode(out),
                max_logits.data_ptr(), max_logits_dims.data(), static_cast<std::uint8_t>(max_logits_dims.size()),
                lse.data_ptr(), lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), lse_indexer.data_ptr(),
                lse_indexer_dims.data(), static_cast<std::uint8_t>(lse_indexer_dims.size()), softmax_scale,
                indexer_topk, (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.sparse_attn_fwd_indexer failed with status ",
                static_cast<unsigned int>(status));
    return {out, max_logits, lse, lse_indexer};
}

at::Tensor sparseAttnTarget(const at::Tensor& q, const at::Tensor& k, const at::Tensor& lse,
                            const at::Tensor& topk_indices, double softmax_scale, bool topk_indices_global) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(k, "k");
    checkDeviceTensor(lse, "lse");
    checkDeviceTensor(topk_indices, "topk_indices");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 3 && lse.dim() == 3 && topk_indices.dim() == 3,
                "sparse_attn_target expects q (B,Sq,Hq,D), k (B,Sk,D), lse (B,Sq,Hq), topk_idxs (B,Sq,topk)");
    at::Tensor target   = torch::empty({q.size(0), q.size(1), topk_indices.size(2)}, q.options().dtype(at::kFloat));
    const auto q_dims   = q.sizes().vec();
    const auto k_dims   = k.sizes().vec();
    const auto lse_dims = lse.sizes().vec();
    const auto topk_indices_dims     = topk_indices.sizes().vec();
    const auto target_dims           = target.sizes().vec();
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (lse).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (target).scalar_type() == at::kFloat) {
            return queryBf16SparseAttnTarget(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), target.data_ptr(), target_dims.data(),
                static_cast<std::uint8_t>(target_dims.size()), softmax_scale, topk_indices_global, &required);
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && (target).scalar_type() == at::kFloat) {
            return queryFp16SparseAttnTarget(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), target.data_ptr(), target_dims.data(),
                static_cast<std::uint8_t>(target_dims.size()), softmax_scale, topk_indices_global, &required);
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && (target).scalar_type() == at::kFloat) {
            return queryFp32SparseAttnTarget(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), target.data_ptr(), target_dims.data(),
                static_cast<std::uint8_t>(target_dims.size()), softmax_scale, topk_indices_global, &required);
        }
        if ((lse).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (target).scalar_type() == at::kFloat) {
            return queryMixedFp32SparseAttnTarget(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), lse.data_ptr(),
                lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), topk_indices.data_ptr(),
                topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()), target.data_ptr(),
                target_dims.data(), static_cast<std::uint8_t>(target_dims.size()), softmax_scale, topk_indices_global,
                &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, target.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (lse).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (target).scalar_type() == at::kFloat) {
            return callBf16SparseAttnTarget(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), target.data_ptr(), target_dims.data(),
                static_cast<std::uint8_t>(target_dims.size()), softmax_scale, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (lse).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && (target).scalar_type() == at::kFloat) {
            return callFp16SparseAttnTarget(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), target.data_ptr(), target_dims.data(),
                static_cast<std::uint8_t>(target_dims.size()), softmax_scale, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (lse).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && (target).scalar_type() == at::kFloat) {
            return callFp32SparseAttnTarget(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), lse.data_ptr(), lse_dims.data(),
                static_cast<std::uint8_t>(lse_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), target.data_ptr(), target_dims.data(),
                static_cast<std::uint8_t>(target_dims.size()), softmax_scale, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((lse).scalar_type() == at::kFloat && (topk_indices).scalar_type() == at::kInt &&
            (target).scalar_type() == at::kFloat) {
            return callMixedFp32SparseAttnTarget(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), lse.data_ptr(),
                lse_dims.data(), static_cast<std::uint8_t>(lse_dims.size()), topk_indices.data_ptr(),
                topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()), target.data_ptr(),
                target_dims.data(), static_cast<std::uint8_t>(target_dims.size()), softmax_scale, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.sparse_attn_target failed with status ",
                static_cast<unsigned int>(status));
    return target;
}

at::Tensor sparseIndexerPredict(const at::Tensor& q, const at::Tensor& k, const at::Tensor& w,
                                const at::Tensor& topk_indices, double sm_scale, bool topk_indices_global) {
    checkDeviceTensor(q, "q");
    // Switch only when needed; RAII restores the caller device on normal and exceptional exits.
    const c10::OptionalDeviceGuard device_guard(c10::impl::VirtualGuardImpl(q.device().type()).getDevice() == q.device()
                                                    ? std::nullopt
                                                    : std::optional<c10::Device>(q.device()));
    checkDeviceTensor(k, "k");
    checkDeviceTensor(w, "w");
    checkDeviceTensor(topk_indices, "topk_indices");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 3 && w.dim() == 3 && topk_indices.dim() == 3,
                "sparse_indexer_predict expects q (B,Sq,Hq,D), k (B,Sk,D), w (B,Sq,Hq), topk_idxs (B,Sq,topk)");
    at::Tensor predict = torch::empty({q.size(0), q.size(1), topk_indices.size(2)}, q.options().dtype(at::kFloat));
    const auto q_dims  = q.sizes().vec();
    const auto k_dims  = k.sizes().vec();
    const auto w_dims  = w.sizes().vec();
    const auto topk_indices_dims = topk_indices.sizes().vec();
    const auto predict_dims      = predict.sizes().vec();

    // The query writes `required`, so it must run before the allocation reads
    // it; argument evaluation order is unspecified, hence two statements.
    std::uint64_t required           = 0;
    const unsigned char query_status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (topk_indices).scalar_type() == at::kInt &&
            (predict).scalar_type() == at::kFloat) {
            return queryBf16SparseIndexerPredict(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), predict.data_ptr(), predict_dims.data(),
                static_cast<std::uint8_t>(predict_dims.size()), sm_scale, topk_indices_global, &required);
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (topk_indices).scalar_type() == at::kInt && (predict).scalar_type() == at::kFloat) {
            return queryFp16SparseIndexerPredict(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), predict.data_ptr(), predict_dims.data(),
                static_cast<std::uint8_t>(predict_dims.size()), sm_scale, topk_indices_global, &required);
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && (predict).scalar_type() == at::kFloat) {
            return queryFp32SparseIndexerPredict(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), predict.data_ptr(), predict_dims.data(),
                static_cast<std::uint8_t>(predict_dims.size()), sm_scale, topk_indices_global, &required);
        }
        if ((topk_indices).scalar_type() == at::kInt && (predict).scalar_type() == at::kFloat) {
            return queryMixedFp32SparseIndexerPredict(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), topk_indices.data_ptr(),
                topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()), predict.data_ptr(),
                predict_dims.data(), static_cast<std::uint8_t>(predict_dims.size()), sm_scale, topk_indices_global,
                &required);
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace =
        torch::empty({static_cast<std::int64_t>(required)}, predict.options().dtype(at::kByte));
    const unsigned char status = [&]() -> unsigned char {
        if ((q).scalar_type() == at::kBFloat16 && (k).scalar_type() == at::kBFloat16 &&
            (w).scalar_type() == at::kBFloat16 && (topk_indices).scalar_type() == at::kInt &&
            (predict).scalar_type() == at::kFloat) {
            return callBf16SparseIndexerPredict(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), predict.data_ptr(), predict_dims.data(),
                static_cast<std::uint8_t>(predict_dims.size()), sm_scale, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kHalf && (k).scalar_type() == at::kHalf && (w).scalar_type() == at::kHalf &&
            (topk_indices).scalar_type() == at::kInt && (predict).scalar_type() == at::kFloat) {
            return callFp16SparseIndexerPredict(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), predict.data_ptr(), predict_dims.data(),
                static_cast<std::uint8_t>(predict_dims.size()), sm_scale, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((q).scalar_type() == at::kFloat && (k).scalar_type() == at::kFloat && (w).scalar_type() == at::kFloat &&
            (topk_indices).scalar_type() == at::kInt && (predict).scalar_type() == at::kFloat) {
            return callFp32SparseIndexerPredict(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), k.data_ptr(), k_dims.data(),
                static_cast<std::uint8_t>(k_dims.size()), w.data_ptr(), w_dims.data(),
                static_cast<std::uint8_t>(w_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
                static_cast<std::uint8_t>(topk_indices_dims.size()), predict.data_ptr(), predict_dims.data(),
                static_cast<std::uint8_t>(predict_dims.size()), sm_scale, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        if ((topk_indices).scalar_type() == at::kInt && (predict).scalar_type() == at::kFloat) {
            return callMixedFp32SparseIndexerPredict(
                q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), k.data_ptr(),
                k_dims.data(), static_cast<std::uint8_t>(k_dims.size()), numericTypeCode(k), w.data_ptr(),
                w_dims.data(), static_cast<std::uint8_t>(w_dims.size()), numericTypeCode(w), topk_indices.data_ptr(),
                topk_indices_dims.data(), static_cast<std::uint8_t>(topk_indices_dims.size()), predict.data_ptr(),
                predict_dims.data(), static_cast<std::uint8_t>(predict_dims.size()), sm_scale, topk_indices_global,
                (required == 0 ? nullptr : workspace.data_ptr()), required, currentStream());
        }
        TORCH_CHECK(false,
                    "flash_train.torch call failed with status UNSUPPORTED: unsupported tensor dtype combination");
        return 0;
    }();
    TORCH_CHECK(status == 0, "flash_train.torch.sparse_indexer_predict failed with status ",
                static_cast<unsigned int>(status));
    return predict;
}

}  // namespace

PYBIND11_MODULE(_flash_train_torch, m) {
    m.doc() = "flash-train PyTorch bindings";
    m.def("gemm", &gemm, "d = alpha * (a @ b) + beta * c on the current stream", py::arg("a"), py::arg("b"),
          py::arg("c") = std::nullopt, py::arg("alpha") = 1.0, py::arg("beta") = 0.0);

    m.def("compactify", &compactify, "Pack each row's valid selections to the front", py::arg("indices"));

    m.def("csa_compressor_bwd", &csaCompressorBwd, "Fused gated-pooling backward", py::arg("kv"), py::arg("score"),
          py::arg("ape"), py::arg("cu_seqlens"), py::arg("cu_seqlens_comp"), py::arg("grad_compressed"),
          py::arg("ratio"), py::arg("head_dim"), py::arg("coff"));

    m.def("csa_compressor_fwd", &csaCompressorFwd, "Fused gated-pooling forward", py::arg("kv"), py::arg("score"),
          py::arg("ape"), py::arg("cu_seqlens"), py::arg("cu_seqlens_comp"), py::arg("ratio"), py::arg("head_dim"),
          py::arg("coff"), py::arg("total_comp"));

    m.def("dense_attn_score", &denseAttnScore, "Dense attention scores and their row total", py::arg("q"), py::arg("k"),
          py::arg("lse"), py::arg("ratio") = 4, py::arg("softmax_scale") = 1.0);

    m.def("dense_attn_score_thd", &denseAttnScoreThd, "Packed-batch dense attention scores", py::arg("q"), py::arg("k"),
          py::arg("lse"), py::arg("cu_seqlens_q"), py::arg("cu_seqlens_kv"), py::arg("ratio") = 4,
          py::arg("softmax_scale") = 1.0, py::arg("max_seqlen_kv") = 0, py::arg("q_causal_offsets") = py::none());

    m.def("dense_indexer_score", &denseIndexerScore, "Dense indexer scores and log-sum-exp denominator", py::arg("q"),
          py::arg("k"), py::arg("w"), py::arg("ratio") = 4, py::arg("sm_scale") = 1.0);

    m.def("dense_indexer_score_thd", &denseIndexerScoreThd, "Packed-batch dense indexer scores and denominator",
          py::arg("q"), py::arg("k"), py::arg("w"), py::arg("cu_seqlens_q"), py::arg("cu_seqlens_kv"),
          py::arg("ratio") = 4, py::arg("sm_scale") = 1.0, py::arg("max_seqlen_kv") = 0,
          py::arg("q_causal_offsets") = py::none());

    m.def("h_aggregate_bwd", &hAggregateBwd, "Aggregation backward", py::arg("grad_aggregated"), py::arg("x"),
          py::arg("h_pre"));

    m.def("h_aggregate_fwd", &hAggregateFwd, "Weighted n-stream aggregation", py::arg("x"), py::arg("h_pre"));

    m.def("h_post_bda_bwd", &hPostBdaBwd,
          "h_post_bda backward; the bias gradient is produced only when the bias is given", py::arg("grad_output"),
          py::arg("residual"), py::arg("x"), py::arg("h_res"), py::arg("h_post"), py::arg("bias") = py::none(),
          py::arg("want_bias_gradient") = true);

    m.def("h_post_bda_fwd", &hPostBdaFwd, "h_res^T @ residual + h_post * (x + bias)", py::arg("h_res"),
          py::arg("residual"), py::arg("h_post"), py::arg("x"), py::arg("bias") = py::none());

    m.def("indexer_bwd_dense", &indexerBwdDense, "Dense indexer backward (updates index_score in place)",
          py::arg("index_q"), py::arg("weights"), py::arg("index_k"), py::arg("attn_score"), py::arg("attn_l1norm"),
          py::arg("index_score"), py::arg("index_lse"), py::arg("sm_scale") = 1.0, py::arg("loss_coeff") = 1.0,
          py::arg("grad_loss") = 1.0, py::arg("ratio") = 4);

    m.def("indexer_bwd_dense_thd", &indexerBwdDenseThd,
          "Packed-batch dense indexer backward (updates index_score in place)", py::arg("index_q"), py::arg("weights"),
          py::arg("index_k"), py::arg("attn_score"), py::arg("attn_l1norm"), py::arg("index_score"),
          py::arg("index_lse"), py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"), py::arg("sm_scale") = 1.0,
          py::arg("loss_coeff") = 1.0, py::arg("grad_loss") = 1.0, py::arg("ratio") = 4,
          py::arg("q_causal_offsets") = py::none());

    m.def("indexer_bwd_sparse", &indexerBwdSparse, "Sparse indexer backward (updates attn_score in place)",
          py::arg("index_q"), py::arg("weights"), py::arg("index_k"), py::arg("attn_score"), py::arg("index_score"),
          py::arg("topk_idxs"), py::arg("sm_scale") = 1.0, py::arg("loss_coeff") = 1.0, py::arg("grad_loss") = 1.0,
          py::arg("topk_indices_global") = false);

    m.def("indexer_fwd", &indexerFwd, "Dense indexer scores under the ratio-causal mask", py::arg("q"), py::arg("k"),
          py::arg("w"), py::arg("ratio") = 4, py::arg("sm_scale") = 1.0);

    m.def("indexer_fwd_thd", &indexerFwdThd, "Packed-batch dense indexer scores", py::arg("q"), py::arg("k"),
          py::arg("w"), py::arg("cu_seqlens_q"), py::arg("cu_seqlens_kv"), py::arg("ratio") = 4,
          py::arg("sm_scale") = 1.0, py::arg("max_seqlen_kv") = 0, py::arg("q_causal_offsets") = py::none());

    m.def("indexer_topk", &indexerTopK, "Per-row top-K selection", py::arg("input_values"), py::arg("seq_lens"),
          py::arg("top_k"), py::arg("next_n") = 1, py::arg("order_values") = true);

    m.def("indexer_topk_thd", &indexerTopKThd, "Packed-batch per-row top-K selection", py::arg("input_values"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_kv"), py::arg("ratio"), py::arg("top_k"),
          py::arg("order_values") = true, py::arg("q_causal_offsets") = py::none());

    m.def("proj_rms_bwd", &projRmsBwd, "Projection and RMS backward", py::arg("grad_proj"), py::arg("grad_r"),
          py::arg("x"), py::arg("weight"), py::arg("r"), py::arg("eps") = 1e-6);

    m.def("proj_rms_compute_h_bwd", &projRmsComputeHBwd,
          "compute_h backward: projection, weight, scalar, and bias gradients", py::arg("grad_h_pre"),
          py::arg("grad_h_post"), py::arg("grad_h_res"), py::arg("grad_r"), py::arg("h_pre"), py::arg("h_post"),
          py::arg("proj"), py::arg("r"), py::arg("x"), py::arg("weight"), py::arg("alpha_pre"), py::arg("alpha_post"),
          py::arg("alpha_res"), py::arg("n"), py::arg("eps") = 1e-6, py::arg("compute_h_eps") = 1e-6);

    m.def("proj_rms_compute_h_fwd", &projRmsComputeHFwd, "Projection, RMS factor, and mHC activation split",
          py::arg("x"), py::arg("weight"), py::arg("alpha_pre"), py::arg("alpha_post"), py::arg("alpha_res"),
          py::arg("bias"), py::arg("n"), py::arg("eps") = 1e-6, py::arg("compute_h_eps") = 1e-6);

    m.def("proj_rms_fwd", &projRmsFwd, "Fused projection plus RMS factor", py::arg("x"), py::arg("weight"),
          py::arg("eps") = 1e-6);

    m.def("sinkhorn_bwd", &sinkhornBwd, "Sinkhorn-Knopp backward", py::arg("grad_output"), py::arg("m_init"),
          py::arg("num_iterations"), py::arg("eps") = 1e-6);

    m.def("sinkhorn_fwd", &sinkhornFwd, "Sinkhorn-Knopp projection plus its FP32 backward state", py::arg("logits"),
          py::arg("num_iterations"), py::arg("eps") = 1e-6);

    m.def("sparse_attn_bwd", &sparseAttnBwd, "Sparse attention backward", py::arg("q"), py::arg("kv"), py::arg("out"),
          py::arg("dout"), py::arg("lse"), py::arg("attn_sink"), py::arg("topk_idxs"), py::arg("softmax_scale") = 1.0,
          py::arg("topk_length") = py::none());

    m.def("sparse_attn_fwd", &sparseAttnFwd, "Sparse attention forward", py::arg("q"), py::arg("kv"),
          py::arg("attn_sink"), py::arg("topk_idxs"), py::arg("softmax_scale") = 1.0,
          py::arg("topk_length") = py::none());

    m.def("sparse_attn_fwd_indexer", &sparseAttnFwdIndexer, "Sparse attention forward with the indexer statistics",
          py::arg("q"), py::arg("kv"), py::arg("attn_sink"), py::arg("topk_idxs"), py::arg("softmax_scale") = 1.0,
          py::arg("indexer_topk") = 0, py::arg("topk_length") = py::none());

    m.def("sparse_attn_target", &sparseAttnTarget, "Normalized teacher scores on the selected entries", py::arg("q"),
          py::arg("k"), py::arg("lse"), py::arg("topk_idxs"), py::arg("softmax_scale") = 1.0,
          py::arg("topk_indices_global") = false);

    m.def("sparse_indexer_predict", &sparseIndexerPredict, "Sparse indexer predictions", py::arg("q"), py::arg("k"),
          py::arg("w"), py::arg("topk_idxs"), py::arg("sm_scale") = 1.0, py::arg("topk_indices_global") = false);
}
