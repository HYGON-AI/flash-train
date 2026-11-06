// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "hygon_distributed_bind.h"

#include <flash_train/engine.hpp>
#include <flash_train/family/ag_group_gemm/family.hpp>
#include <flash_train/family/moe_reduce_rs/family.hpp>
#include <flash_train/operation/distributed.hpp>
#include <flash_train/ops_args.hpp>
#include <flash_train/plan.hpp>
#include <flash_train/pattern.hpp>
// Include the implementation directly so the pybind registration TU exports
// the session factory symbols.  Hidden-visibility linking across the two
// objects confuses the HIP helper's incremental object list.
#include "../compat/distributed_session.cpp"
#include <ths_op/hygon_shm.h>
#include <flash_train/handle.hpp>
#include <flash_train/session/distributed.hpp>

#include <torch/csrc/distributed/c10d/PyProcessGroup.hpp>

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPFunctions.h>

#include <pybind11/stl.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flash_train_torch {

ftrain::AGGroupGemmAttributes agAttributes(std::uint64_t session_id) {
    const auto session = ftrain::DistributedSessionRegistry::get().getAGGroupGemm(session_id);
    TORCH_CHECK(session != nullptr, "AGGroupGemm session was not found");
    return ftrain::AGGroupGemmAttributes(session->info().route, session_id);
}

ftrain::MoEReduceRSAttributes moeAttributes(std::uint64_t session_id) {
    const auto session = ftrain::DistributedSessionRegistry::get().getMoEReduceRS(session_id);
    TORCH_CHECK(session != nullptr, "MoEReduceRS session was not found");
    return ftrain::MoEReduceRSAttributes(session->info().route, session->info().gemm_variant, session_id);
}

namespace {

c10::intrusive_ptr<c10d::ProcessGroup> processGroupFromPython(const py::object& value) {
    auto* instance = reinterpret_cast<pybind11::detail::instance*>(value.ptr());
    TORCH_CHECK(instance->simple_layout, "ProcessGroup must use pybind simple layout");
    auto* raw = static_cast<c10d::ProcessGroup*>(instance->simple_value_holder[0]);
    TORCH_CHECK(raw != nullptr, "ProcessGroup pybind instance has no C++ value");
    return c10::intrusive_ptr<c10d::ProcessGroup>::reclaim_copy(raw);
}

FTrainNumericType numericTypeFromAt(c10::ScalarType value) {
    switch (value) {
        case at::kHalf: return FTRAIN_NUMERIC_TYPE_FP16;
        case at::kBFloat16: return FTRAIN_NUMERIC_TYPE_BF16;
        case at::kInt: return FTRAIN_NUMERIC_TYPE_INT32;
        case at::kLong: return FTRAIN_NUMERIC_TYPE_INT64;
        default: TORCH_CHECK(false, "Hygon distributed operators require FP16/BF16 and INT32/INT64 routing");
    }
}

ftrain::StorageView storageViewFromAt(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "Hygon distributed tensors must be contiguous device tensors");
    FTrainStorageView view{};
    view.memory       = tensor.data_ptr();
    view.dims        = tensor.sizes().data();
    view.strides     = tensor.strides().data();
    view.num_dims    = static_cast<std::uint8_t>(tensor.dim());
    view.numeric_type = numericTypeFromAt(tensor.scalar_type());
    view.index_type  = FTRAIN_INDEX_TYPE_INVALID;
    view.is_host_memory = false;
    return ftrain::StorageView(view);
}

void destroyPlan(PyObject* capsule) {
    auto* plan = static_cast<ftrain::Plan*>(PyCapsule_GetPointer(capsule, "FTrainPlan"));
    delete plan;
}

ftrain::Plan& planFromCapsule(const py::capsule& capsule) {
    if (!capsule || std::string(capsule.name()) != "FTrainPlan") {
        throw std::invalid_argument("invalid FTrainPlan capsule");
    }
    return *static_cast<ftrain::Plan*>(capsule.get_pointer());
}

py::capsule createAGPlan(std::uint64_t session_id, const at::Tensor& input, const at::Tensor& weight,
                         const at::Tensor& topk_ids, const at::Tensor& output) {
    const auto& roles = ftrain::AGGroupGemmFamily::makeRoles();
    ftrain::PatternBuilder user;
    const auto input_id = user.addOperand<ftrain::OperandKind::kTensor>();
    const auto weight_id = user.addOperand<ftrain::OperandKind::kTensor>();
    const auto topk_id = user.addOperand<ftrain::OperandKind::kTensor>();
    const auto output_id = user.addOperand<ftrain::OperandKind::kTensor>();
    const auto op_id = user.addOperation<ftrain::OperationKind::kAGGroupGemm>(input_id, weight_id, topk_id, output_id);
    ftrain::Ops ops(user.buildPattern(), roles.pattern);
    ftrain::Args args = ops.makeArgs();
    args.setOperand<ftrain::OperandKind::kTensor>(
        ftrain::OperandTraits<ftrain::OperandKind::kTensor>::createPatternOperandId(input_id),
        ftrain::Tensor(storageViewFromAt(input)));
    args.setOperand<ftrain::OperandKind::kTensor>(
        ftrain::OperandTraits<ftrain::OperandKind::kTensor>::createPatternOperandId(weight_id),
        ftrain::Tensor(storageViewFromAt(weight)));
    args.setOperand<ftrain::OperandKind::kTensor>(
        ftrain::OperandTraits<ftrain::OperandKind::kTensor>::createPatternOperandId(topk_id),
        ftrain::Tensor(storageViewFromAt(topk_ids)));
    args.setOperand<ftrain::OperandKind::kTensor>(
        ftrain::OperandTraits<ftrain::OperandKind::kTensor>::createPatternOperandId(output_id),
        ftrain::Tensor(storageViewFromAt(output)));
    args.setOperation<ftrain::OperationKind::kAGGroupGemm>(
        ftrain::OperationTraits<ftrain::OperationKind::kAGGroupGemm>::createPatternOperationId(op_id),
        agAttributes(session_id));
    auto engine = ftrain::getGlobalHandle().findOpsEngine(ops.getPatternKey());
    TORCH_CHECK(engine != nullptr, "AGGroupGemm formal engine is not registered");
    return py::capsule(new ftrain::Plan(engine->createPlan(args)), "FTrainPlan", destroyPlan);
}

py::capsule createMoEPlan(std::uint64_t session_id, const at::Tensor& input, const at::Tensor& weight,
                          const at::Tensor& chosen_experts, const at::Tensor& expert_weight,
                          const at::Tensor& output) {
    const auto& roles = ftrain::MoEReduceRSFamily::makeRoles();
    ftrain::PatternBuilder user;
    const auto input_id = user.addOperand<ftrain::OperandKind::kTensor>();
    const auto weight_id = user.addOperand<ftrain::OperandKind::kTensor>();
    const auto expert_id = user.addOperand<ftrain::OperandKind::kTensor>();
    const auto scale_id = user.addOperand<ftrain::OperandKind::kTensor>();
    const auto output_id = user.addOperand<ftrain::OperandKind::kTensor>();
    const auto op_id = user.addOperation<ftrain::OperationKind::kMoEReduceRS>(
        input_id, weight_id, expert_id, scale_id, output_id);
    ftrain::Ops ops(user.buildPattern(), roles.pattern);
    ftrain::Args args = ops.makeArgs();
    args.setOperand<ftrain::OperandKind::kTensor>(
        ftrain::OperandTraits<ftrain::OperandKind::kTensor>::createPatternOperandId(input_id),
        ftrain::Tensor(storageViewFromAt(input)));
    args.setOperand<ftrain::OperandKind::kTensor>(
        ftrain::OperandTraits<ftrain::OperandKind::kTensor>::createPatternOperandId(weight_id),
        ftrain::Tensor(storageViewFromAt(weight)));
    args.setOperand<ftrain::OperandKind::kTensor>(
        ftrain::OperandTraits<ftrain::OperandKind::kTensor>::createPatternOperandId(expert_id),
        ftrain::Tensor(storageViewFromAt(chosen_experts)));
    args.setOperand<ftrain::OperandKind::kTensor>(
        ftrain::OperandTraits<ftrain::OperandKind::kTensor>::createPatternOperandId(scale_id),
        ftrain::Tensor(storageViewFromAt(expert_weight)));
    args.setOperand<ftrain::OperandKind::kTensor>(
        ftrain::OperandTraits<ftrain::OperandKind::kTensor>::createPatternOperandId(output_id),
        ftrain::Tensor(storageViewFromAt(output)));
    args.setOperation<ftrain::OperationKind::kMoEReduceRS>(
        ftrain::OperationTraits<ftrain::OperationKind::kMoEReduceRS>::createPatternOperationId(op_id),
        moeAttributes(session_id));
    auto engine = ftrain::getGlobalHandle().findOpsEngine(ops.getPatternKey());
    TORCH_CHECK(engine != nullptr, "MoEReduceRS formal engine is not registered");
    return py::capsule(new ftrain::Plan(engine->createPlan(args)), "FTrainPlan", destroyPlan);
}

}  // namespace

void bindHygonDistributed(py::module_& module) {
    auto initHygonShm = [](py::object etp_group) {
          return bytedance::hygon_vendor::init_hygon_shm(processGroupFromPython(etp_group));
        };
    module.def("init_hygon_shm", initHygonShm, py::arg("etp_group"));

    module.def("create_ag_group_gemm_session", [](py::object etp_group, int32_t nnodes, int32_t full_m,
                                                     int32_t n_dim, int32_t k_dim, int32_t num_experts, int32_t topk,
                                                     py::object dtype, std::vector<int64_t> rank_m_sizes,
                                                     int32_t capacity_full_m, std::string route) {
          return ftrain::compat::createAGGroupGemmSession(
              std::move(route), processGroupFromPython(etp_group), nnodes, full_m, n_dim, k_dim, num_experts, topk,
              torch::python::detail::py_object_to_dtype(dtype), std::move(rank_m_sizes), capacity_full_m);
        },
        py::arg("etp_group"), py::arg("nnodes"), py::arg("full_m"), py::arg("n_dim"), py::arg("k_dim"),
        py::arg("num_experts"), py::arg("topk"), py::arg("dtype"), py::arg("rank_m_sizes") = std::vector<int64_t>{},
        py::arg("capacity_full_m") = -1, py::arg("route") = "native_nk_a2a");

    module.def("close_ag_group_gemm_session", [](std::uint64_t session_id) {
          ftrain::DistributedSessionRegistry::get().eraseAGGroupGemm(session_id);
        },
        py::arg("session_id"));

    module.def("create_ag_group_gemm_plan", [](std::uint64_t session_id, const at::Tensor& input,
                                                  const at::Tensor& weight, const at::Tensor& topk_ids,
                                                  const at::Tensor& output) {
          return createAGPlan(session_id, input, weight, topk_ids, output);
        },
        py::arg("session_id"), py::arg("input"), py::arg("weight"), py::arg("topk_ids"), py::arg("output"));

    module.def("create_moe_reduce_rs_session", [](py::object etp_group, int32_t nnodes, int32_t max_ntokens,
                                                     int32_t n_dim, int32_t k_dim, int32_t num_experts, int32_t topk,
                                                     py::object dtype, std::vector<int64_t> rank_m_sizes,
                                                     int32_t n_chunks, std::string route,
                                                     std::string gemm_variant) {
          return ftrain::compat::createMoEReduceRSSession(
              std::move(route), std::move(gemm_variant), processGroupFromPython(etp_group), nnodes, max_ntokens,
              n_dim, k_dim, num_experts, topk, torch::python::detail::py_object_to_dtype(dtype),
              std::move(rank_m_sizes), n_chunks);
        },
        py::arg("etp_group"), py::arg("nnodes"), py::arg("max_ntokens"), py::arg("n_dim"), py::arg("k_dim"),
        py::arg("num_experts"), py::arg("topk"), py::arg("dtype"), py::arg("rank_m_sizes") = std::vector<int64_t>{},
        py::arg("n_chunks") = 0, py::arg("route") = "a2a_ragged_signal",
        py::arg("gemm_variant") = "nk_direct");

    module.def("close_moe_reduce_rs_session", [](std::uint64_t session_id) {
          ftrain::DistributedSessionRegistry::get().eraseMoEReduceRS(session_id);
        },
        py::arg("session_id"));

    module.def("create_moe_reduce_rs_plan", [](std::uint64_t session_id, const at::Tensor& input,
                                                  const at::Tensor& weight, const at::Tensor& chosen_experts,
                                                  const at::Tensor& expert_weight, const at::Tensor& output) {
          return createMoEPlan(session_id, input, weight, chosen_experts, expert_weight, output);
        },
        py::arg("session_id"), py::arg("input"), py::arg("weight"), py::arg("chosen_experts"),
        py::arg("expert_weight"), py::arg("output"));

    module.def("get_plan_num_primitives", [](const py::capsule& plan) {
          return planFromCapsule(plan).getNumPrimitives();
        },
        py::arg("plan"));

    module.def("get_plan_primitive_name", [](const py::capsule& plan, std::uint64_t index) {
          return planFromCapsule(plan).getPrimitiveName(index);
        },
        py::arg("plan"), py::arg("index"));

    module.def("execute_plan", [](const py::capsule& plan, std::uint64_t index) {
          auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
          planFromCapsule(plan).execute(index, nullptr, 0, stream);
        },
        py::arg("plan"), py::arg("index") = 0);
}

}  // namespace flash_train_torch
