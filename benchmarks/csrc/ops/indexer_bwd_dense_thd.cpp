// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <hip/hip_fp16.h>

#include <flash_train/flash_train.h>

#include "op.hpp"

namespace ftrain_bench {
namespace {

struct IndexerBwdDenseThdSession final : OpSession {
    IndexerBwdDenseThdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        auto a = shape[0], b = shape[1], c = shape[2];
        std::int64_t batch = 1, kv_tokens = a, top_k = std::min<std::int64_t>(a, 32), ratio = 1;
        std::vector<std::int64_t> segments{a};
        if (shape[0] == 8192 && shape[1] == 64 && shape[2] == 128) {
            segments = {4096, 4096};
            ratio    = 4;
        }
        if (shape[0] == 12288 && shape[1] == 8 && shape[2] == 128) {
            segments = {4096, 8192};
            ratio    = 4;
        }
        if (shape[0] == 32768 && shape[1] == 64 && shape[2] == 128) {
            segments = {16384, 16384};
            ratio    = 128;
        }
        if (shape[0] == 8192 && shape[1] == 32 && shape[2] == 128) {
            segments = {4096, 4096};
            ratio    = 1;
        }
        if (shape[0] == 12288 && shape[1] == 4 && shape[2] == 128) {
            segments = {4096, 8192};
            ratio    = 1;
        }
        std::vector<std::int64_t> cu_q{0}, cu_k{0};
        std::int64_t max_k = 0;
        for (auto length : segments) {
            cu_q.push_back(cu_q.back() + length);
            cu_k.push_back(cu_k.back() + length / ratio);
            max_k = std::max(max_k, length / ratio);
        }
        kv_tokens = cu_k.back();
        std::array<FTrainNumericType, 18> types{};
        switch (compute) {
            case FTRAIN_NUMERIC_TYPE_BF16:
                types = {FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_BF16,
                         FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_BF16,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT64};
                break;
            case FTRAIN_NUMERIC_TYPE_FP16:
                types = {FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP16,
                         FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP16,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT64};
                break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                types = {FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT32,
                         FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,
                         FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_NUMERIC_TYPE_INT64};
                break;
            default: require(false, "unsupported precision");
        }
        index_q_dims_ = {a, b, c};
        const std::size_t index_q_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t index_q_bytes = index_q_count * elementBytes(types[0]);
        host_.resize((index_q_bytes + 7) / 8);
        for (std::size_t j = 0; j < index_q_count; ++j) { setHostValue(j, 0.25, types[0]); }
        checkHip(hipMalloc(&dindex_q_, index_q_bytes), "malloc index_q");
        checkHip(hipMemcpy(dindex_q_, host_.data(), index_q_bytes, hipMemcpyHostToDevice), "fill index_q");
        view_index_q_                   = dense(dindex_q_, index_q_dims_, types[0]);
        weights_dims_                   = {a, b};
        const std::size_t weights_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t weights_bytes = weights_count * elementBytes(types[1]);
        host_.resize((weights_bytes + 7) / 8);
        for (std::size_t j = 0; j < weights_count; ++j) { setHostValue(j, 1, types[1]); }
        checkHip(hipMalloc(&dweights_, weights_bytes), "malloc weights");
        checkHip(hipMemcpy(dweights_, host_.data(), weights_bytes, hipMemcpyHostToDevice), "fill weights");
        view_weights_                   = dense(dweights_, weights_dims_, types[1]);
        index_k_dims_                   = {kv_tokens, c};
        const std::size_t index_k_count = static_cast<std::size_t>(kv_tokens) * static_cast<std::size_t>(c);
        const std::size_t index_k_bytes = index_k_count * elementBytes(types[2]);
        host_.resize((index_k_bytes + 7) / 8);
        for (std::size_t j = 0; j < index_k_count; ++j) { setHostValue(j, 0.5, types[2]); }
        checkHip(hipMalloc(&dindex_k_, index_k_bytes), "malloc index_k");
        checkHip(hipMemcpy(dindex_k_, host_.data(), index_k_bytes, hipMemcpyHostToDevice), "fill index_k");
        view_index_k_                      = dense(dindex_k_, index_k_dims_, types[2]);
        attn_score_dims_                   = {a, max_k};
        const std::size_t attn_score_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(max_k);
        const std::size_t attn_score_bytes = attn_score_count * elementBytes(types[3]);
        host_.resize((attn_score_bytes + 7) / 8);
        for (std::size_t j = 0; j < attn_score_count; ++j) { setHostValue(j, 1, types[3]); }
        checkHip(hipMalloc(&dattn_score_, attn_score_bytes), "malloc attn_score");
        checkHip(hipMemcpy(dattn_score_, host_.data(), attn_score_bytes, hipMemcpyHostToDevice), "fill attn_score");
        view_attn_score_                    = dense(dattn_score_, attn_score_dims_, types[3]);
        attn_l1norm_dims_                   = {a};
        const std::size_t attn_l1norm_count = static_cast<std::size_t>(a);
        const std::size_t attn_l1norm_bytes = attn_l1norm_count * elementBytes(types[4]);
        host_.resize((attn_l1norm_bytes + 7) / 8);
        for (std::size_t j = 0; j < attn_l1norm_count; ++j) { setHostValue(j, j + 1, types[4]); }
        checkHip(hipMalloc(&dattn_l1norm_, attn_l1norm_bytes), "malloc attn_l1norm");
        checkHip(hipMemcpy(dattn_l1norm_, host_.data(), attn_l1norm_bytes, hipMemcpyHostToDevice), "fill attn_l1norm");
        view_attn_l1norm_                   = dense(dattn_l1norm_, attn_l1norm_dims_, types[4]);
        index_score_dims_                   = {a, max_k};
        const std::size_t index_score_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(max_k);
        const std::size_t index_score_bytes = index_score_count * elementBytes(types[5]);
        host_.resize((index_score_bytes + 7) / 8);
        for (std::size_t j = 0; j < index_score_count; ++j) { setHostValue(j, 2, types[5]); }
        checkHip(hipMalloc(&dindex_score_, index_score_bytes), "malloc index_score");
        checkHip(hipMemcpy(dindex_score_, host_.data(), index_score_bytes, hipMemcpyHostToDevice), "fill index_score");
        view_index_score_                 = dense(dindex_score_, index_score_dims_, types[5]);
        index_lse_dims_                   = {a};
        const std::size_t index_lse_count = static_cast<std::size_t>(a);
        const std::size_t index_lse_bytes = index_lse_count * elementBytes(types[6]);
        host_.resize((index_lse_bytes + 7) / 8);
        for (std::size_t j = 0; j < index_lse_count; ++j) { setHostValue(j, 2.0 + std::log(j + 1.0), types[6]); }
        checkHip(hipMalloc(&dindex_lse_, index_lse_bytes), "malloc index_lse");
        checkHip(hipMemcpy(dindex_lse_, host_.data(), index_lse_bytes, hipMemcpyHostToDevice), "fill index_lse");
        view_index_lse_                      = dense(dindex_lse_, index_lse_dims_, types[6]);
        cu_seqlens_q_dims_                   = {static_cast<std::int64_t>(segments.size() + 1)};
        const std::size_t cu_seqlens_q_count = static_cast<std::size_t>(static_cast<std::int64_t>(segments.size() + 1));
        const std::size_t cu_seqlens_q_bytes = cu_seqlens_q_count * elementBytes(types[7]);
        host_.resize((cu_seqlens_q_bytes + 7) / 8);
        for (std::size_t j = 0; j < cu_seqlens_q_count; ++j) { setHostValue(j, 0, types[7]); }
        for (std::size_t j = 0; j < cu_q.size(); ++j) { setHostValue(j, cu_q[j], types[7]); }
        checkHip(hipMalloc(&dcu_seqlens_q_, cu_seqlens_q_bytes), "malloc cu_seqlens_q");
        checkHip(hipMemcpy(dcu_seqlens_q_, host_.data(), cu_seqlens_q_bytes, hipMemcpyHostToDevice),
                 "fill cu_seqlens_q");
        view_cu_seqlens_q_                   = dense(dcu_seqlens_q_, cu_seqlens_q_dims_, types[7]);
        cu_seqlens_k_dims_                   = {static_cast<std::int64_t>(segments.size() + 1)};
        const std::size_t cu_seqlens_k_count = static_cast<std::size_t>(static_cast<std::int64_t>(segments.size() + 1));
        const std::size_t cu_seqlens_k_bytes = cu_seqlens_k_count * elementBytes(types[8]);
        host_.resize((cu_seqlens_k_bytes + 7) / 8);
        for (std::size_t j = 0; j < cu_seqlens_k_count; ++j) { setHostValue(j, 0, types[8]); }
        for (std::size_t j = 0; j < cu_k.size(); ++j) { setHostValue(j, cu_k[j], types[8]); }
        checkHip(hipMalloc(&dcu_seqlens_k_, cu_seqlens_k_bytes), "malloc cu_seqlens_k");
        checkHip(hipMemcpy(dcu_seqlens_k_, host_.data(), cu_seqlens_k_bytes, hipMemcpyHostToDevice),
                 "fill cu_seqlens_k");
        view_cu_seqlens_k_                       = dense(dcu_seqlens_k_, cu_seqlens_k_dims_, types[8]);
        q_causal_offsets_dims_                   = {static_cast<std::int64_t>(segments.size())};
        const std::size_t q_causal_offsets_count = static_cast<std::size_t>(static_cast<std::int64_t>(segments.size()));
        const std::size_t q_causal_offsets_bytes = q_causal_offsets_count * elementBytes(types[9]);
        host_.resize((q_causal_offsets_bytes + 7) / 8);
        for (std::size_t j = 0; j < q_causal_offsets_count; ++j) { setHostValue(j, 0, types[9]); }
        checkHip(hipMalloc(&dq_causal_offsets_, q_causal_offsets_bytes), "malloc q_causal_offsets");
        checkHip(hipMemcpy(dq_causal_offsets_, host_.data(), q_causal_offsets_bytes, hipMemcpyHostToDevice),
                 "fill q_causal_offsets");
        view_q_causal_offsets_                   = dense(dq_causal_offsets_, q_causal_offsets_dims_, types[9]);
        // The primitive requires score gradients to alias the input score buffer.
        view_grad_index_score_        = view_index_score_;
        dq_dims_                      = {a, b, c};
        const std::size_t dq_count =
            static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t dq_bytes = dq_count * elementBytes(types[11]);
        host_.resize((dq_bytes + 7) / 8);
        for (std::size_t j = 0; j < dq_count; ++j) { setHostValue(j, 123, types[11]); }
        checkHip(hipMalloc(&ddq_, dq_bytes), "malloc dq");
        checkHip(hipMemcpy(ddq_, host_.data(), dq_bytes, hipMemcpyHostToDevice), "fill dq");
        view_dq_                   = dense(ddq_, dq_dims_, types[11]);
        dw_dims_                   = {a, b};
        const std::size_t dw_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
        const std::size_t dw_bytes = dw_count * elementBytes(types[12]);
        host_.resize((dw_bytes + 7) / 8);
        for (std::size_t j = 0; j < dw_count; ++j) { setHostValue(j, 123, types[12]); }
        checkHip(hipMalloc(&ddw_, dw_bytes), "malloc dw");
        checkHip(hipMemcpy(ddw_, host_.data(), dw_bytes, hipMemcpyHostToDevice), "fill dw");
        view_dw_                   = dense(ddw_, dw_dims_, types[12]);
        dk_dims_                   = {kv_tokens, c};
        const std::size_t dk_count = static_cast<std::size_t>(kv_tokens) * static_cast<std::size_t>(c);
        const std::size_t dk_bytes = dk_count * elementBytes(types[13]);
        host_.resize((dk_bytes + 7) / 8);
        for (std::size_t j = 0; j < dk_count; ++j) { setHostValue(j, 123, types[13]); }
        checkHip(hipMalloc(&ddk_, dk_bytes), "malloc dk");
        checkHip(hipMemcpy(ddk_, host_.data(), dk_bytes, hipMemcpyHostToDevice), "fill dk");
        view_dk_              = dense(ddk_, dk_dims_, types[13]);
        sm_scale_value_       = static_cast<float>(0.25);
        view_sm_scale_        = scalar(&sm_scale_value_, types[14]);
        loss_coeff_value_     = static_cast<float>(1);
        view_loss_coeff_      = scalar(&loss_coeff_value_, types[15]);
        grad_loss_value_      = static_cast<float>(1);
        view_grad_loss_       = scalar(&grad_loss_value_, types[16]);
        ratio_value_          = static_cast<std::int64_t>(ratio);
        view_ratio_           = scalar(&ratio_value_, types[17]);
        FTrainPattern pattern = nullptr;
        FTrainOps ops         = nullptr;
        FTrainArgs args       = nullptr;
        FTrainPlan plan       = nullptr;
        checkFtrain(ftrainPatternCreate(&pattern), "workspace pattern");
        buildPattern(pattern);
        checkFtrain(ftrainOpsCreate(&ops, pattern), "workspace ops");
        checkFtrain(ftrainArgsCreate(&args, ops), "workspace args");
        bindArgs(args);
        checkFtrain(ftrainPlanCreate(&plan, args), "workspace plan");
        checkFtrain(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &workspace_bytes_), "workspace size");
        if (workspace_bytes_) { checkHip(hipMalloc(&workspace_, workspace_bytes_), "workspace allocation"); }
        ftrainPlanDestroy(plan);
        ftrainArgsDestroy(args);
        ftrainOpsDestroy(ops);
        ftrainPatternDestroy(pattern);
    }
    ~IndexerBwdDenseThdSession() override {
        hipFree(dindex_q_);
        hipFree(dweights_);
        hipFree(dindex_k_);
        hipFree(dattn_score_);
        hipFree(dattn_l1norm_);
        hipFree(dindex_score_);
        hipFree(dindex_lse_);
        hipFree(dcu_seqlens_q_);
        hipFree(dcu_seqlens_k_);
        hipFree(dq_causal_offsets_);
        hipFree(ddq_);
        hipFree(ddw_);
        hipFree(ddk_);
        if (workspace_) { hipFree(workspace_); }
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainIndexerBwdDenseThd(
                        view_index_q_, view_weights_, view_index_k_, view_attn_score_, view_attn_l1norm_,
                        view_index_score_, view_index_lse_, view_cu_seqlens_q_, view_cu_seqlens_k_,
                        view_q_causal_offsets_, view_grad_index_score_, view_dq_, view_dw_, view_dk_, view_sm_scale_,
                        view_loss_coeff_, view_grad_loss_, view_ratio_, compute_, workspace_, workspace_bytes_, stream),
                    "ftrainIndexerBwdDenseThd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tindex_q_), "add index_q");
        checkFtrain(ftrainPatternAddTensor(pattern, &tweights_), "add weights");
        checkFtrain(ftrainPatternAddTensor(pattern, &tindex_k_), "add index_k");
        checkFtrain(ftrainPatternAddTensor(pattern, &tattn_score_), "add attn_score");
        checkFtrain(ftrainPatternAddTensor(pattern, &tattn_l1norm_), "add attn_l1norm");
        checkFtrain(ftrainPatternAddTensor(pattern, &tindex_score_), "add index_score");
        checkFtrain(ftrainPatternAddTensor(pattern, &tindex_lse_), "add index_lse");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcu_seqlens_q_), "add cu_seqlens_q");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcu_seqlens_k_), "add cu_seqlens_k");
        checkFtrain(ftrainPatternAddTensor(pattern, &tq_causal_offsets_), "add q_causal_offsets");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_index_score_), "add grad_index_score");
        checkFtrain(ftrainPatternAddTensor(pattern, &tdq_), "add dq");
        checkFtrain(ftrainPatternAddTensor(pattern, &tdw_), "add dw");
        checkFtrain(ftrainPatternAddTensor(pattern, &tdk_), "add dk");
        checkFtrain(ftrainPatternAddTensor(pattern, &tsm_scale_), "add sm_scale");
        checkFtrain(ftrainPatternAddTensor(pattern, &tloss_coeff_), "add loss_coeff");
        checkFtrain(ftrainPatternAddTensor(pattern, &tgrad_loss_), "add grad_loss");
        checkFtrain(ftrainPatternAddTensor(pattern, &tratio_), "add ratio");
        checkFtrain(ftrainPatternAddIndexerBwdDenseThd(
                        pattern, &operation_, tindex_q_, tweights_, tindex_k_, tattn_score_, tattn_l1norm_,
                        tindex_score_, tindex_lse_, tcu_seqlens_q_, tcu_seqlens_k_, tq_causal_offsets_,
                        tgrad_index_score_, tdq_, tdw_, tdk_, tsm_scale_, tloss_coeff_, tgrad_loss_, tratio_),
                    "ftrainPatternAddIndexerBwdDenseThd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tindex_q_, view_index_q_), "set index_q");
        checkFtrain(ftrainArgsSetTensor(args, tweights_, view_weights_), "set weights");
        checkFtrain(ftrainArgsSetTensor(args, tindex_k_, view_index_k_), "set index_k");
        checkFtrain(ftrainArgsSetTensor(args, tattn_score_, view_attn_score_), "set attn_score");
        checkFtrain(ftrainArgsSetTensor(args, tattn_l1norm_, view_attn_l1norm_), "set attn_l1norm");
        checkFtrain(ftrainArgsSetTensor(args, tindex_score_, view_index_score_), "set index_score");
        checkFtrain(ftrainArgsSetTensor(args, tindex_lse_, view_index_lse_), "set index_lse");
        checkFtrain(ftrainArgsSetTensor(args, tcu_seqlens_q_, view_cu_seqlens_q_), "set cu_seqlens_q");
        checkFtrain(ftrainArgsSetTensor(args, tcu_seqlens_k_, view_cu_seqlens_k_), "set cu_seqlens_k");
        checkFtrain(ftrainArgsSetTensor(args, tq_causal_offsets_, view_q_causal_offsets_), "set q_causal_offsets");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_index_score_, view_grad_index_score_), "set grad_index_score");
        checkFtrain(ftrainArgsSetTensor(args, tdq_, view_dq_), "set dq");
        checkFtrain(ftrainArgsSetTensor(args, tdw_, view_dw_), "set dw");
        checkFtrain(ftrainArgsSetTensor(args, tdk_, view_dk_), "set dk");
        checkFtrain(ftrainArgsSetTensor(args, tsm_scale_, view_sm_scale_), "set sm_scale");
        checkFtrain(ftrainArgsSetTensor(args, tloss_coeff_, view_loss_coeff_), "set loss_coeff");
        checkFtrain(ftrainArgsSetTensor(args, tgrad_loss_, view_grad_loss_), "set grad_loss");
        checkFtrain(ftrainArgsSetTensor(args, tratio_, view_ratio_), "set ratio");
        checkFtrain(ftrainArgsSetIndexerBwdDenseThd(args, operation_, compute_), "ftrainArgsSetIndexerBwdDenseThd");
    }

  private:
    static FTrainStorageView dense(void* memory, const std::vector<std::int64_t>& dims, FTrainNumericType type) {
        return {
            memory, dims.data(), nullptr, static_cast<std::uint8_t>(dims.size()), type, FTRAIN_INDEX_TYPE_CONTINUOUS,
            false};
    }
    static FTrainStorageView scalar(void* value, FTrainNumericType type) {
        return {value, nullptr, nullptr, 0, type, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    }
    static std::size_t elementBytes(FTrainNumericType type) {
        return (type == FTRAIN_NUMERIC_TYPE_INT64)
                   ? 8
                   : ((type == FTRAIN_NUMERIC_TYPE_BF16 || type == FTRAIN_NUMERIC_TYPE_FP16) ? 2 : (4));
    }
    void setHostValue(std::size_t index, double value, FTrainNumericType type) {
        auto* dst = reinterpret_cast<unsigned char*>(host_.data()) + index * elementBytes(type);
        if (type == FTRAIN_NUMERIC_TYPE_FP32) {
            const float v = static_cast<float>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_FP16) {
            const __half v = __float2half(static_cast<float>(value));
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_BF16) {
            const float v = static_cast<float>(value);
            std::uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            bits                        += 0x7fff + ((bits >> 16) & 1);
            const std::uint16_t rounded  = static_cast<std::uint16_t>(bits >> 16);
            std::memcpy(dst, &rounded, sizeof(rounded));
        } else if (type == FTRAIN_NUMERIC_TYPE_INT64) {
            const std::int64_t v = static_cast<std::int64_t>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else if (type == FTRAIN_NUMERIC_TYPE_INT32) {
            const std::int32_t v = static_cast<std::int32_t>(value);
            std::memcpy(dst, &v, sizeof(v));
        } else {
            require(false, "unsupported benchmark port type");
        }
    }
    FTrainNumericType compute_;
    std::vector<std::int64_t> index_q_dims_;
    void* dindex_q_ = nullptr;
    std::vector<std::int64_t> weights_dims_;
    void* dweights_ = nullptr;
    std::vector<std::int64_t> index_k_dims_;
    void* dindex_k_ = nullptr;
    std::vector<std::int64_t> attn_score_dims_;
    void* dattn_score_ = nullptr;
    std::vector<std::int64_t> attn_l1norm_dims_;
    void* dattn_l1norm_ = nullptr;
    std::vector<std::int64_t> index_score_dims_;
    void* dindex_score_ = nullptr;
    std::vector<std::int64_t> index_lse_dims_;
    void* dindex_lse_ = nullptr;
    std::vector<std::int64_t> cu_seqlens_q_dims_;
    void* dcu_seqlens_q_ = nullptr;
    std::vector<std::int64_t> cu_seqlens_k_dims_;
    void* dcu_seqlens_k_ = nullptr;
    std::vector<std::int64_t> q_causal_offsets_dims_;
    void* dq_causal_offsets_ = nullptr;
    std::vector<std::int64_t> dq_dims_;
    void* ddq_ = nullptr;
    std::vector<std::int64_t> dw_dims_;
    void* ddw_ = nullptr;
    std::vector<std::int64_t> dk_dims_;
    void* ddk_ = nullptr;
    float sm_scale_value_{};
    float loss_coeff_value_{};
    float grad_loss_value_{};
    std::int64_t ratio_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_index_q_{};
    FTrainTensorId tindex_q_{};
    FTrainStorageView view_weights_{};
    FTrainTensorId tweights_{};
    FTrainStorageView view_index_k_{};
    FTrainTensorId tindex_k_{};
    FTrainStorageView view_attn_score_{};
    FTrainTensorId tattn_score_{};
    FTrainStorageView view_attn_l1norm_{};
    FTrainTensorId tattn_l1norm_{};
    FTrainStorageView view_index_score_{};
    FTrainTensorId tindex_score_{};
    FTrainStorageView view_index_lse_{};
    FTrainTensorId tindex_lse_{};
    FTrainStorageView view_cu_seqlens_q_{};
    FTrainTensorId tcu_seqlens_q_{};
    FTrainStorageView view_cu_seqlens_k_{};
    FTrainTensorId tcu_seqlens_k_{};
    FTrainStorageView view_q_causal_offsets_{};
    FTrainTensorId tq_causal_offsets_{};
    FTrainStorageView view_grad_index_score_{};
    FTrainTensorId tgrad_index_score_{};
    FTrainStorageView view_dq_{};
    FTrainTensorId tdq_{};
    FTrainStorageView view_dw_{};
    FTrainTensorId tdw_{};
    FTrainStorageView view_dk_{};
    FTrainTensorId tdk_{};
    FTrainStorageView view_sm_scale_{};
    FTrainTensorId tsm_scale_{};
    FTrainStorageView view_loss_coeff_{};
    FTrainTensorId tloss_coeff_{};
    FTrainStorageView view_grad_loss_{};
    FTrainTensorId tgrad_loss_{};
    FTrainStorageView view_ratio_{};
    FTrainTensorId tratio_{};
    FTrainIndexerBwdDenseThdOpId operation_{};
    void* workspace_               = nullptr;
    std::uint64_t workspace_bytes_ = 0;
};
struct IndexerBwdDenseThdCase final : OpCase {
    const char* name() const override { return "indexer_bwd_dense_thd"; }
    const char* shapeGrammar() const override { return "tokens,heads,head_dim"; }
    const char* convenienceImpl() const override { return "ftrainIndexerBwdDenseThd"; }
    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {
            {"bf16", FTRAIN_NUMERIC_TYPE_BF16},
            {"fp16", FTRAIN_NUMERIC_TYPE_FP16},
            {"fp32", FTRAIN_NUMERIC_TYPE_FP32}
        };
    }
    std::unique_ptr<OpSession> create(const std::vector<std::int64_t>& shape,
                                      FTrainNumericType compute) const override {
        require(shape.size() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "tokens,heads,head_dim: three positive values required");
        return std::make_unique<IndexerBwdDenseThdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<IndexerBwdDenseThdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
