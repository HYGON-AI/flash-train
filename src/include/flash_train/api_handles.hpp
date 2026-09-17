#ifndef FTRAIN_API_HANDLES_HPP_
#define FTRAIN_API_HANDLES_HPP_

#include <memory>
#include <utility>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/primitive.hpp"
#include "flash_train/runtime.hpp"

struct FTrainPatternStruct final {
    ftrain::Pattern pattern;
};

struct FTrainOpsStruct final {
    explicit FTrainOpsStruct(ftrain::Ops&& ops_operand) noexcept : ops(std::move(ops_operand)) {}

    ftrain::Ops ops;
};

struct FTrainArgsStruct final {
    FTrainArgsStruct(const ftrain::Ops& ops, const ftrain::SupportedSchema& supported_schema)
        : args(ops, supported_schema) {}

    ftrain::Args args;
};

struct FTrainPlanStruct final {
    FTrainPlanStruct(FTrainDeviceId plan_device_id, std::unique_ptr<ftrain::PrimitiveBase>&& primitive_operand)
        : device_id(plan_device_id) {
        primitives.push_back(std::move(primitive_operand));
    }

    FTrainPlanStruct(FTrainDeviceId plan_device_id,
                     std::vector<std::unique_ptr<ftrain::PrimitiveBase>>&& primitive_operands)
        : device_id(plan_device_id), primitives(std::move(primitive_operands)) {}

    FTrainDeviceId device_id;
    std::vector<std::unique_ptr<ftrain::PrimitiveBase>> primitives;
};

#endif
