// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace flash_train_torch {
void* prepareGdnPlan(int kind, const void* const* memory, const std::int64_t* const* dims,
                     const std::uint8_t* ranks, const std::uint8_t* types, float scale,
                     std::uint64_t* workspace_bytes, unsigned char* result);
unsigned char executeGdnPlan(void* plan, void* workspace, std::uint64_t workspace_bytes, void* stream);
void destroyGdnPlan(void* plan);
const char* lastGdnMessage();
}  // namespace flash_train_torch

namespace {

struct Shape {
    std::int64_t b, s, h, hv, k, v;
};

void* currentStream() {
    py::object cuda = py::module_::import("torch").attr("cuda");
    return reinterpret_cast<void*>(py::cast<std::uintptr_t>(cuda.attr("current_stream")().attr("cuda_stream")));
}

Shape checkQkv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool fp32_allowed) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 4, "q must be a rank-four device tensor");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 || (fp32_allowed && q.scalar_type() == at::kFloat),
                "q dtype is unsupported");
    TORCH_CHECK(v.dim() == 4, "v must have rank four");
    const Shape shape{q.size(0), q.size(1), q.size(2), v.size(2), q.size(3), v.size(3)};
    TORCH_CHECK(shape.b > 0 && shape.s > 0 && shape.h > 0 && shape.hv > 0 &&
                shape.k > 0 && shape.v > 0 && shape.hv % shape.h == 0, "GDN dimensions are invalid");
    TORCH_CHECK(q.get_device() == py::cast<int>(py::module_::import("torch").attr("cuda").attr("current_device")()),
                "q must be on the current device");
    return shape;
}

void checkTensor(const at::Tensor& tensor, const char* name, at::ScalarType type,
                 const std::vector<std::int64_t>& dims, const at::Tensor& q) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.device() == q.device(),
                name, " must be on the same device as q");
    TORCH_CHECK(tensor.scalar_type() == type, name, " dtype is unsupported");
    TORCH_CHECK(tensor.sizes().vec() == dims, name, " shape is invalid");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void checkOptionalState(const std::optional<at::Tensor>& state, const char* name,
                        const Shape& shape, const at::Tensor& q) {
    if (state) checkTensor(*state, name, at::kFloat, {shape.b, shape.hv, shape.k, shape.v}, q);
}

float getScale(std::optional<double> scale, const Shape& shape) {
    const double selected = scale.value_or(1.0 / std::sqrt(static_cast<double>(shape.k)));
    TORCH_CHECK(std::isfinite(selected) && selected > 0 && selected <= std::numeric_limits<float>::max(),
                "scale must be finite and positive");
    return static_cast<float>(selected);
}

struct Plan {
    void* handle = nullptr;
    ~Plan() { flash_train_torch::destroyGdnPlan(handle); }
};

void run(int kind, const std::vector<at::Tensor>& ports, const at::Tensor& q, float scale) {
    std::array<const void*, 14> memory{};
    std::array<const std::int64_t*, 14> dims{};
    std::array<std::array<std::int64_t, 4>, 14> empty{};
    std::array<std::uint8_t, 14> ranks{}, types{};
    for (std::size_t i = 0; i < ports.size(); ++i) {
        const auto& tensor = ports[i];
        if (tensor.defined()) {
            TORCH_CHECK(tensor.is_cuda() && tensor.device() == q.device() && tensor.is_contiguous(),
                        "GDN port ", i, " must be contiguous on q's device");
            TORCH_CHECK(tensor.dim() <= 4, "GDN port ", i, " has invalid rank");
            TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kBFloat16,
                        "GDN port ", i, " has invalid dtype");
            memory[i] = tensor.data_ptr();
            dims[i] = tensor.sizes().data();
            ranks[i] = static_cast<std::uint8_t>(tensor.dim());
            types[i] = tensor.scalar_type() == at::kFloat ? 4 : 8;
        } else {
            empty[i] = {0, ports[2].size(2), q.size(3), ports[2].size(3)};
            dims[i] = empty[i].data();
            ranks[i] = 4;
            types[i] = 4;
        }
    }
    std::uint64_t required = 0;
    unsigned char status = 0;
    Plan plan{flash_train_torch::prepareGdnPlan(kind, memory.data(), dims.data(), ranks.data(), types.data(),
                                               scale, &required, &status)};
    TORCH_CHECK(status == 0 && plan.handle, "GDN Plan: ", flash_train_torch::lastGdnMessage());
    TORCH_CHECK(required <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
                "GDN workspace size is too large");
    at::Tensor workspace;
    if (required) workspace = torch::empty({static_cast<std::int64_t>(required)}, q.options().dtype(at::kByte));
    status = flash_train_torch::executeGdnPlan(plan.handle, required ? workspace.data_ptr() : nullptr,
                                                required, currentStream());
    TORCH_CHECK(status == 0, "GDN execution: ", flash_train_torch::lastGdnMessage());
}

std::tuple<at::Tensor, std::optional<at::Tensor>> gdnForward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
    const at::Tensor& beta, std::optional<double> scale, std::optional<at::Tensor> initial_state,
    bool output_final_state) {
    const Shape shape = checkQkv(q, k, v, true);
    const auto type = q.scalar_type();
    checkTensor(q, "q", type, {shape.b, shape.s, shape.h, shape.k}, q);
    checkTensor(k, "k", type, {shape.b, shape.s, shape.h, shape.k}, q);
    checkTensor(v, "v", type, {shape.b, shape.s, shape.hv, shape.v}, q);
    checkTensor(g, "g", at::kFloat, {shape.b, shape.s, shape.hv}, q);
    checkTensor(beta, "beta", at::kFloat, {shape.b, shape.s, shape.hv}, q);
    checkOptionalState(initial_state, "initial_state", shape, q);
    auto out = torch::empty({shape.b, shape.s, shape.hv, shape.v}, q.options());
    std::optional<at::Tensor> ht;
    if (output_final_state) ht = torch::empty({shape.b, shape.hv, shape.k, shape.v}, q.options().dtype(at::kFloat));
    run(0, {q, k, v, g, beta, initial_state.value_or(at::Tensor{}), out, ht.value_or(at::Tensor{})},
        q, getScale(scale, shape));
    return {out, ht};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, std::optional<at::Tensor>> gdnBackward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
    const at::Tensor& beta, const at::Tensor& dout, std::optional<double> scale,
    std::optional<at::Tensor> h0, std::optional<at::Tensor> dht, bool need_dh0) {
    const Shape shape = checkQkv(q, k, v, false);
    checkTensor(q, "q", at::kBFloat16, {shape.b, shape.s, shape.h, shape.k}, q);
    checkTensor(k, "k", at::kBFloat16, {shape.b, shape.s, shape.h, shape.k}, q);
    checkTensor(v, "v", at::kBFloat16, {shape.b, shape.s, shape.hv, shape.v}, q);
    checkTensor(g, "g", at::kFloat, {shape.b, shape.s, shape.hv}, q);
    checkTensor(beta, "beta", at::kFloat, {shape.b, shape.s, shape.hv}, q);
    checkTensor(dout, "do", at::kBFloat16, {shape.b, shape.s, shape.hv, shape.v}, q);
    checkOptionalState(h0, "h0", shape, q);
    checkOptionalState(dht, "dht", shape, q);
    auto dq = torch::empty_like(q), dk = torch::empty_like(k), dv = torch::empty_like(v);
    auto dg = torch::empty_like(g), dbeta = torch::empty_like(beta);
    std::optional<at::Tensor> dh0;
    if (need_dh0) dh0 = torch::empty({shape.b, shape.hv, shape.k, shape.v}, q.options().dtype(at::kFloat));
    run(1, {q, k, v, g, beta, dout, h0.value_or(at::Tensor{}), dht.value_or(at::Tensor{}),
            dq, dk, dv, dg, dbeta, dh0.value_or(at::Tensor{})}, q, getScale(scale, shape));
    return {dq, dk, dv, dg, dbeta, dh0};
}

std::tuple<at::Tensor, std::optional<at::Tensor>> gdn2Forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
    const at::Tensor& b, const at::Tensor& w, std::optional<double> scale) {
    const Shape shape = checkQkv(q, k, v, false);
    checkTensor(q, "q", at::kBFloat16, {shape.b, shape.s, shape.h, shape.k}, q);
    checkTensor(k, "k", at::kBFloat16, {shape.b, shape.s, shape.h, shape.k}, q);
    checkTensor(v, "v", at::kBFloat16, {shape.b, shape.s, shape.hv, shape.v}, q);
    checkTensor(g, "g", at::kFloat, {shape.b, shape.s, shape.hv, shape.k}, q);
    checkTensor(b, "b", at::kFloat, {shape.b, shape.s, shape.hv, shape.k}, q);
    checkTensor(w, "w", at::kFloat, {shape.b, shape.s, shape.hv, shape.v}, q);
    auto out = torch::empty_like(v);
    run(2, {q, k, v, g, b, w, out}, q, getScale(scale, shape));
    return {out, std::nullopt};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> gdn2Backward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
    const at::Tensor& b, const at::Tensor& w, const at::Tensor& dout, std::optional<double> scale) {
    const Shape shape = checkQkv(q, k, v, false);
    checkTensor(q, "q", at::kBFloat16, {shape.b, shape.s, shape.h, shape.k}, q);
    checkTensor(k, "k", at::kBFloat16, {shape.b, shape.s, shape.h, shape.k}, q);
    checkTensor(v, "v", at::kBFloat16, {shape.b, shape.s, shape.hv, shape.v}, q);
    checkTensor(g, "g", at::kFloat, {shape.b, shape.s, shape.hv, shape.k}, q);
    checkTensor(b, "b", at::kFloat, {shape.b, shape.s, shape.hv, shape.k}, q);
    checkTensor(w, "w", at::kFloat, {shape.b, shape.s, shape.hv, shape.v}, q);
    checkTensor(dout, "do", at::kBFloat16, {shape.b, shape.s, shape.hv, shape.v}, q);
    auto dq = torch::empty_like(q), dk = torch::empty_like(k), dv = torch::empty_like(v);
    auto dg = torch::empty_like(g);
    auto db = torch::empty(b.sizes(), q.options()), dw = torch::empty(w.sizes(), q.options());
    run(3, {q, k, v, g, b, w, dout, dq, dk, dv, dg, db, dw}, q, getScale(scale, shape));
    return {dq, dk, dv, dg, db, dw};
}

}  // namespace

void bindGdn(py::module_& m) {
    m.def("gdn_forward", &gdnForward, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
          py::arg("beta"), py::arg("scale") = std::nullopt, py::arg("initial_state") = std::nullopt,
          py::arg("output_final_state") = false);
    m.def("gdn_backward", &gdnBackward, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
          py::arg("beta"), py::arg("do"), py::arg("scale") = std::nullopt, py::arg("h0") = std::nullopt,
          py::arg("dht") = std::nullopt, py::arg("need_dh0") = false);
    m.def("gdn2_forward", &gdn2Forward, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
          py::arg("b"), py::arg("w"), py::arg("scale") = std::nullopt);
    m.def("gdn2_backward", &gdn2Backward, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
          py::arg("b"), py::arg("w"), py::arg("do"), py::arg("scale") = std::nullopt);
}
