// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

namespace flash_train_torch {

// Defined in flash_train_torch_capi.cpp. Plain pointer and integer types keep
// Torch's CUDA-compat headers separate from the HIP runtime headers.
std::uint8_t numericTypeCodeFp32();
std::uint8_t numericTypeCodeFp16();
std::uint8_t numericTypeCodeBf16();
std::uint8_t numericTypeCodeInt32();

unsigned char callSparseAttnBwd(
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
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream, std::uint8_t compute_type);

unsigned char querySparseAttnBwd(
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
    double softmax_scale, std::uint64_t* workspace_bytes, std::uint8_t compute_type);

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

// Native implementations use a uniform floating dtype. Mixed storage uses FP32
// computation; operators without a mixed implementation require uniform storage.
std::uint8_t computeTypeCode(std::initializer_list<at::ScalarType> types, bool allow_mixed = true) {
    const auto first = *types.begin();
    bool uniform     = true;
    for (const auto type : types) {
        TORCH_CHECK(type == at::kFloat || type == at::kHalf || type == at::kBFloat16,
                    "flash_train.torch: unsupported floating tensor dtype ", type);
        uniform = uniform && type == first;
    }
    TORCH_CHECK(allow_mixed || uniform, "flash_train.torch: unsupported mixed tensor dtypes");
    if (uniform && first == at::kBFloat16) { return numericTypeCodeBf16(); }
    if (uniform && first == at::kHalf) { return numericTypeCodeFp16(); }
    return numericTypeCodeFp32();
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
    for (const auto* tensor : {&kv, &out, &dout, &lse, &attn_sink, &topk_indices}) {
        TORCH_CHECK(tensor->device() == q.device(), "sparse_attn_bwd tensors must share q's device");
    }
    if (topk_length.has_value()) {
        checkDeviceTensor(*topk_length, "topk_length");
        TORCH_CHECK(topk_length->device() == q.device(), "topk_length must share q's device");
    }

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

    std::uint64_t required = 0;
    TORCH_CHECK((lse).scalar_type() == at::kFloat && (attn_sink).scalar_type() == at::kFloat &&
                    (topk_indices).scalar_type() == at::kInt && topk_length_type == at::kInt &&
                    (dsink).scalar_type() == at::kFloat,
                "flash_train.torch: unsupported tensor dtype combination");
    // Initial operator PR: only the BF16 baseline is registered.
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16 &&
                    out.scalar_type() == at::kBFloat16 && dout.scalar_type() == at::kBFloat16,
                "sparse_attn_bwd currently supports BF16 storage only");
    const std::uint8_t compute_type  = numericTypeCodeBf16();
    const unsigned char query_status = querySparseAttnBwd(
        q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), kv.data_ptr(),
        kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), numericTypeCode(kv), out.data_ptr(), out_dims.data(),
        static_cast<std::uint8_t>(out_dims.size()), numericTypeCode(out), dout.data_ptr(), dout_dims.data(),
        static_cast<std::uint8_t>(dout_dims.size()), numericTypeCode(dout), lse.data_ptr(), lse_dims.data(),
        static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
        static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
        static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
        static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(), dq_dims.data(),
        static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dkv.data_ptr(), dkv_dims.data(),
        static_cast<std::uint8_t>(dkv_dims.size()), numericTypeCode(dkv), dsink.data_ptr(), dsink_dims.data(),
        static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale, &required, compute_type);
    TORCH_CHECK(query_status == 0, "flash_train.torch workspace query failed with status ",
                static_cast<unsigned int>(query_status));
    const at::Tensor workspace = torch::empty({static_cast<std::int64_t>(required)}, dq.options().dtype(at::kByte));
    const unsigned char status = callSparseAttnBwd(
        q.data_ptr(), q_dims.data(), static_cast<std::uint8_t>(q_dims.size()), numericTypeCode(q), kv.data_ptr(),
        kv_dims.data(), static_cast<std::uint8_t>(kv_dims.size()), numericTypeCode(kv), out.data_ptr(), out_dims.data(),
        static_cast<std::uint8_t>(out_dims.size()), numericTypeCode(out), dout.data_ptr(), dout_dims.data(),
        static_cast<std::uint8_t>(dout_dims.size()), numericTypeCode(dout), lse.data_ptr(), lse_dims.data(),
        static_cast<std::uint8_t>(lse_dims.size()), attn_sink.data_ptr(), attn_sink_dims.data(),
        static_cast<std::uint8_t>(attn_sink_dims.size()), topk_indices.data_ptr(), topk_indices_dims.data(),
        static_cast<std::uint8_t>(topk_indices_dims.size()), topk_length_data, topk_length_dims.data(),
        static_cast<std::uint8_t>(topk_length_dims.size()), dq.data_ptr(), dq_dims.data(),
        static_cast<std::uint8_t>(dq_dims.size()), numericTypeCode(dq), dkv.data_ptr(), dkv_dims.data(),
        static_cast<std::uint8_t>(dkv_dims.size()), numericTypeCode(dkv), dsink.data_ptr(), dsink_dims.data(),
        static_cast<std::uint8_t>(dsink_dims.size()), softmax_scale, (required == 0 ? nullptr : workspace.data_ptr()),
        required, currentStream(), compute_type);
    TORCH_CHECK(status == 0, "flash_train.torch.sparse_attn_bwd failed with status ",
                static_cast<unsigned int>(status));
    return {dq, dkv, dsink};
}

}  // namespace

PYBIND11_MODULE(_flash_train_torch, m) {
    m.doc() = "flash-train PyTorch bindings";
    m.def("gemm", &gemm, "d = alpha * (a @ b) + beta * c on the current stream", py::arg("a"), py::arg("b"),
          py::arg("c") = std::nullopt, py::arg("alpha") = 1.0, py::arg("beta") = 0.0);

    m.def("sparse_attn_bwd", &sparseAttnBwd, "Sparse attention backward", py::arg("q"), py::arg("kv"), py::arg("out"),
          py::arg("dout"), py::arg("lse"), py::arg("attn_sink"), py::arg("topk_idxs"), py::arg("softmax_scale") = 1.0,
          py::arg("topk_length") = py::none());
}
