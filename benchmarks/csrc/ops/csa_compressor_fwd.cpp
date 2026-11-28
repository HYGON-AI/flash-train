// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <flash_train/flash_train.h>

#include "op.hpp"

namespace ftrain_bench {
namespace {

struct CSACompressorFwdSession final : OpSession {
    CSACompressorFwdSession(const std::vector<std::int64_t>& shape, FTrainNumericType compute) : compute_(compute) {
        auto a = shape[0], b = shape[1], c = shape[2];
        std::int64_t batch = 1, kv_tokens = a, top_k = std::min<std::int64_t>(a, 32), ratio = 1;
        std::int64_t coff = 1, head_dim = c;
        if (shape[0] == 4096 && shape[1] == 4 && shape[2] == 256) {
            coff     = 2;
            head_dim = 128;
        }
        if (shape[0] == 4096 && shape[1] == 4 && shape[2] == 1024) {
            coff     = 2;
            head_dim = 512;
        }
        if (shape[0] == 4096 && shape[1] == 128 && shape[2] == 128) {
            coff     = 1;
            head_dim = 128;
        }
        if (shape[0] == 4096 && shape[1] == 128 && shape[2] == 512) {
            coff     = 1;
            head_dim = 512;
        }
        if (shape[0] == 8192 && shape[1] == 4 && shape[2] == 256) {
            coff     = 2;
            head_dim = 128;
        }
        if (shape[0] == 8192 && shape[1] == 4 && shape[2] == 1024) {
            coff     = 2;
            head_dim = 512;
        }
        if (shape[0] == 8192 && shape[1] == 128 && shape[2] == 128) {
            coff     = 1;
            head_dim = 128;
        }
        if (shape[0] == 8192 && shape[1] == 128 && shape[2] == 512) {
            coff     = 1;
            head_dim = 512;
        }
        if (shape[0] == 16384 && shape[1] == 4 && shape[2] == 256) {
            coff     = 2;
            head_dim = 128;
        }
        if (shape[0] == 16384 && shape[1] == 4 && shape[2] == 1024) {
            coff     = 2;
            head_dim = 512;
        }
        if (shape[0] == 16384 && shape[1] == 128 && shape[2] == 128) {
            coff     = 1;
            head_dim = 128;
        }
        if (shape[0] == 16384 && shape[1] == 128 && shape[2] == 512) {
            coff     = 1;
            head_dim = 512;
        }
        require(compute == FTRAIN_NUMERIC_TYPE_BF16, "unsupported precision");
        kv_dims_                   = {a, c};
        const std::size_t kv_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(c);
        const std::size_t kv_bytes = kv_count * elementBytes(FTRAIN_NUMERIC_TYPE_BF16);
        host_.resize((kv_bytes + 7) / 8);
        for (std::size_t j = 0; j < kv_count; ++j) { setHostValue(j, 0.5, FTRAIN_NUMERIC_TYPE_BF16); }
        checkHip(hipMalloc(&dkv_, kv_bytes), "malloc kv");
        checkHip(hipMemcpy(dkv_, host_.data(), kv_bytes, hipMemcpyHostToDevice), "fill kv");
        view_kv_                      = dense(dkv_, kv_dims_, FTRAIN_NUMERIC_TYPE_BF16);
        score_dims_                   = {a, c};
        const std::size_t score_count = static_cast<std::size_t>(a) * static_cast<std::size_t>(c);
        const std::size_t score_bytes = score_count * elementBytes(FTRAIN_NUMERIC_TYPE_BF16);
        host_.resize((score_bytes + 7) / 8);
        for (std::size_t j = 0; j < score_count; ++j) { setHostValue(j, 0, FTRAIN_NUMERIC_TYPE_BF16); }
        checkHip(hipMalloc(&dscore_, score_bytes), "malloc score");
        checkHip(hipMemcpy(dscore_, host_.data(), score_bytes, hipMemcpyHostToDevice), "fill score");
        view_score_                 = dense(dscore_, score_dims_, FTRAIN_NUMERIC_TYPE_BF16);
        ape_dims_                   = {b, c};
        const std::size_t ape_count = static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
        const std::size_t ape_bytes = ape_count * elementBytes(FTRAIN_NUMERIC_TYPE_FP32);
        host_.resize((ape_bytes + 7) / 8);
        for (std::size_t j = 0; j < ape_count; ++j) { setHostValue(j, 0, FTRAIN_NUMERIC_TYPE_FP32); }
        checkHip(hipMalloc(&dape_, ape_bytes), "malloc ape");
        checkHip(hipMemcpy(dape_, host_.data(), ape_bytes, hipMemcpyHostToDevice), "fill ape");
        view_ape_                          = dense(dape_, ape_dims_, FTRAIN_NUMERIC_TYPE_FP32);
        cu_seqlens_dims_                   = {2};
        const std::size_t cu_seqlens_count = static_cast<std::size_t>(2);
        const std::size_t cu_seqlens_bytes = cu_seqlens_count * elementBytes(FTRAIN_NUMERIC_TYPE_INT32);
        host_.resize((cu_seqlens_bytes + 7) / 8);
        for (std::size_t j = 0; j < cu_seqlens_count; ++j) { setHostValue(j, 0, FTRAIN_NUMERIC_TYPE_INT32); }
        setHostValue(0, 0, FTRAIN_NUMERIC_TYPE_INT32);
        setHostValue(1, a, FTRAIN_NUMERIC_TYPE_INT32);
        checkHip(hipMalloc(&hcu_seqlens_, cu_seqlens_bytes), "malloc cu_seqlens");
        checkHip(hipMemcpy(hcu_seqlens_, host_.data(), cu_seqlens_bytes, hipMemcpyHostToDevice), "fill cu_seqlens");
        view_cu_seqlens_                        = dense(hcu_seqlens_, cu_seqlens_dims_, FTRAIN_NUMERIC_TYPE_INT32);
        cu_seqlens_comp_dims_                   = {2};
        const std::size_t cu_seqlens_comp_count = static_cast<std::size_t>(2);
        const std::size_t cu_seqlens_comp_bytes = cu_seqlens_comp_count * elementBytes(FTRAIN_NUMERIC_TYPE_INT32);
        host_.resize((cu_seqlens_comp_bytes + 7) / 8);
        for (std::size_t j = 0; j < cu_seqlens_comp_count; ++j) { setHostValue(j, 0, FTRAIN_NUMERIC_TYPE_INT32); }
        setHostValue(0, 0, FTRAIN_NUMERIC_TYPE_INT32);
        setHostValue(1, a / b, FTRAIN_NUMERIC_TYPE_INT32);
        checkHip(hipMalloc(&hcu_seqlens_comp_, cu_seqlens_comp_bytes), "malloc cu_seqlens_comp");
        checkHip(hipMemcpy(hcu_seqlens_comp_, host_.data(), cu_seqlens_comp_bytes, hipMemcpyHostToDevice),
                 "fill cu_seqlens_comp");
        view_cu_seqlens_comp_              = dense(hcu_seqlens_comp_, cu_seqlens_comp_dims_, FTRAIN_NUMERIC_TYPE_INT32);
        compressed_dims_                   = {a / b, head_dim};
        const std::size_t compressed_count = static_cast<std::size_t>(a / b) * static_cast<std::size_t>(head_dim);
        const std::size_t compressed_bytes = compressed_count * elementBytes(FTRAIN_NUMERIC_TYPE_BF16);
        host_.resize((compressed_bytes + 7) / 8);
        for (std::size_t j = 0; j < compressed_count; ++j) { setHostValue(j, 123, FTRAIN_NUMERIC_TYPE_BF16); }
        checkHip(hipMalloc(&dcompressed_, compressed_bytes), "malloc compressed");
        checkHip(hipMemcpy(dcompressed_, host_.data(), compressed_bytes, hipMemcpyHostToDevice), "fill compressed");
        view_compressed_  = dense(dcompressed_, compressed_dims_, FTRAIN_NUMERIC_TYPE_BF16);
        ratio_value_      = static_cast<std::int64_t>(b);
        view_ratio_       = scalar(&ratio_value_, FTRAIN_NUMERIC_TYPE_INT64);
        head_dim_value_   = static_cast<std::int64_t>(head_dim);
        view_head_dim_    = scalar(&head_dim_value_, FTRAIN_NUMERIC_TYPE_INT64);
        coff_value_       = static_cast<std::int64_t>(coff);
        view_coff_        = scalar(&coff_value_, FTRAIN_NUMERIC_TYPE_INT64);
        total_comp_value_ = static_cast<std::int64_t>(a / b);
        view_total_comp_  = scalar(&total_comp_value_, FTRAIN_NUMERIC_TYPE_INT64);
    }
    ~CSACompressorFwdSession() override {
        hipFree(dkv_);
        hipFree(dscore_);
        hipFree(dape_);
        hipFree(hcu_seqlens_);
        hipFree(hcu_seqlens_comp_);
        hipFree(dcompressed_);
    }
    void runConvenience(hipStream_t stream) override {
        checkFtrain(ftrainCSACompressorFwd(view_kv_, view_score_, view_ape_, view_cu_seqlens_, view_cu_seqlens_comp_,
                                           view_compressed_, view_ratio_, view_head_dim_, view_coff_, view_total_comp_,
                                           compute_, nullptr, 0, stream),
                    "ftrainCSACompressorFwd");
    }
    void buildPattern(FTrainPattern pattern) override {
        checkFtrain(ftrainPatternAddTensor(pattern, &tkv_), "add kv");
        checkFtrain(ftrainPatternAddTensor(pattern, &tscore_), "add score");
        checkFtrain(ftrainPatternAddTensor(pattern, &tape_), "add ape");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcu_seqlens_), "add cu_seqlens");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcu_seqlens_comp_), "add cu_seqlens_comp");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcompressed_), "add compressed");
        checkFtrain(ftrainPatternAddTensor(pattern, &tratio_), "add ratio");
        checkFtrain(ftrainPatternAddTensor(pattern, &thead_dim_), "add head_dim");
        checkFtrain(ftrainPatternAddTensor(pattern, &tcoff_), "add coff");
        checkFtrain(ftrainPatternAddTensor(pattern, &ttotal_comp_), "add total_comp");
        checkFtrain(ftrainPatternAddCSACompressorFwd(pattern, &operation_, tkv_, tscore_, tape_, tcu_seqlens_,
                                                     tcu_seqlens_comp_, tcompressed_, tratio_, thead_dim_, tcoff_,
                                                     ttotal_comp_),
                    "ftrainPatternAddCSACompressorFwd");
    }
    void bindArgs(FTrainArgs args) override {
        checkFtrain(ftrainArgsSetTensor(args, tkv_, view_kv_), "set kv");
        checkFtrain(ftrainArgsSetTensor(args, tscore_, view_score_), "set score");
        checkFtrain(ftrainArgsSetTensor(args, tape_, view_ape_), "set ape");
        checkFtrain(ftrainArgsSetTensor(args, tcu_seqlens_, view_cu_seqlens_), "set cu_seqlens");
        checkFtrain(ftrainArgsSetTensor(args, tcu_seqlens_comp_, view_cu_seqlens_comp_), "set cu_seqlens_comp");
        checkFtrain(ftrainArgsSetTensor(args, tcompressed_, view_compressed_), "set compressed");
        checkFtrain(ftrainArgsSetTensor(args, tratio_, view_ratio_), "set ratio");
        checkFtrain(ftrainArgsSetTensor(args, thead_dim_, view_head_dim_), "set head_dim");
        checkFtrain(ftrainArgsSetTensor(args, tcoff_, view_coff_), "set coff");
        checkFtrain(ftrainArgsSetTensor(args, ttotal_comp_, view_total_comp_), "set total_comp");
        checkFtrain(ftrainArgsSetCSACompressorFwd(args, operation_, compute_), "ftrainArgsSetCSACompressorFwd");
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
        return (type == FTRAIN_NUMERIC_TYPE_INT64) ? 8 : ((type == FTRAIN_NUMERIC_TYPE_BF16) ? 2 : (4));
    }
    void setHostValue(std::size_t index, double value, FTrainNumericType type) {
        auto* dst = reinterpret_cast<unsigned char*>(host_.data()) + index * elementBytes(type);
        if (type == FTRAIN_NUMERIC_TYPE_FP32) {
            const float v = static_cast<float>(value);
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
    std::vector<std::int64_t> kv_dims_;
    void* dkv_ = nullptr;
    std::vector<std::int64_t> score_dims_;
    void* dscore_ = nullptr;
    std::vector<std::int64_t> ape_dims_;
    void* dape_ = nullptr;
    std::vector<std::int64_t> cu_seqlens_dims_;
    void* hcu_seqlens_ = nullptr;
    std::vector<std::int64_t> cu_seqlens_comp_dims_;
    void* hcu_seqlens_comp_ = nullptr;
    std::vector<std::int64_t> compressed_dims_;
    void* dcompressed_ = nullptr;
    std::int64_t ratio_value_{};
    std::int64_t head_dim_value_{};
    std::int64_t coff_value_{};
    std::int64_t total_comp_value_{};
    std::vector<std::uint64_t> host_;
    FTrainStorageView view_kv_{};
    FTrainTensorId tkv_{};
    FTrainStorageView view_score_{};
    FTrainTensorId tscore_{};
    FTrainStorageView view_ape_{};
    FTrainTensorId tape_{};
    FTrainStorageView view_cu_seqlens_{};
    FTrainTensorId tcu_seqlens_{};
    FTrainStorageView view_cu_seqlens_comp_{};
    FTrainTensorId tcu_seqlens_comp_{};
    FTrainStorageView view_compressed_{};
    FTrainTensorId tcompressed_{};
    FTrainStorageView view_ratio_{};
    FTrainTensorId tratio_{};
    FTrainStorageView view_head_dim_{};
    FTrainTensorId thead_dim_{};
    FTrainStorageView view_coff_{};
    FTrainTensorId tcoff_{};
    FTrainStorageView view_total_comp_{};
    FTrainTensorId ttotal_comp_{};
    FTrainCSACompressorFwdOpId operation_{};
};
struct CSACompressorFwdCase final : OpCase {
    const char* name() const override { return "csa_compressor_fwd"; }
    const char* shapeGrammar() const override { return "tokens,ratio,input_width"; }
    const char* convenienceImpl() const override { return "ftrainCSACompressorFwd"; }
    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {
            {"bf16", FTRAIN_NUMERIC_TYPE_BF16}
        };
    }
    std::unique_ptr<OpSession> create(const std::vector<std::int64_t>& shape,
                                      FTrainNumericType compute) const override {
        require(shape.size() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "tokens,ratio,input_width: three positive values required");
        return std::make_unique<CSACompressorFwdSession>(shape, compute);
    }
};
const bool kRegistered = [] {
    registerOp(std::make_unique<CSACompressorFwdCase>());
    return true;
}();
}  // namespace
}  // namespace ftrain_bench
