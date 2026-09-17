#include "flash_train/binding.hpp"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

#include "flash_train/error.hpp"

namespace ftrain {
namespace {

constexpr std::size_t kUnmappedRole = std::numeric_limits<std::size_t>::max();

OperandKind getOperandKind(const OpsOperand& operand) noexcept {
    return std::visit(
        [](const auto& typed_operand) noexcept {
            using OperandType = std::decay_t<decltype(typed_operand)>;
            return OpsOperandTraits<OperandType>::kKind;
        },
        operand);
}

OpsOperand makeUnsetOperand(OperandKind kind) {
    switch (kind) {
        case OperandKind::kTensor: return Tensor{};
        case OperandKind::kTensorList: return TensorList{};
        case OperandKind::kGroupedTensor: return GroupedTensor{};
    }
    throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Unsupported OperandKind %u in supported schema",
                    static_cast<unsigned int>(kind));
}

bool isOperandSet(const OpsOperand& operand) noexcept {
    return std::visit([](const auto& typed_operand) noexcept { return typed_operand.isSet(); }, operand);
}

}  // namespace

RoleMapping RoleMapping::fromMatchResult(const MatchResult& result) {
    const std::size_t num_operands = result.getNumMappedOperands();
    const std::size_t num_ops      = result.getNumMappedOps();

    std::vector<std::size_t> supported_operand_indices_by_user_operand(num_operands, 0);
    std::vector<std::size_t> user_operand_indices_by_supported_operand(num_operands, 0);
    for (std::size_t user_operand_index = 0; user_operand_index < num_operands; ++user_operand_index) {
        const std::size_t supported_operand_index =
            result.getSupportedOperandId(PatternOperandId{user_operand_index}).getIndex();
        supported_operand_indices_by_user_operand[user_operand_index]      = supported_operand_index;
        user_operand_indices_by_supported_operand[supported_operand_index] = user_operand_index;
    }

    std::vector<std::size_t> supported_op_indices_by_user_op(num_ops, 0);
    std::vector<std::size_t> user_op_indices_by_supported_op(num_ops, 0);
    for (std::size_t user_op_index = 0; user_op_index < num_ops; ++user_op_index) {
        const std::size_t supported_op_index = result.getSupportedOpId(PatternOperationId{user_op_index}).getIndex();
        supported_op_indices_by_user_op[user_op_index]      = supported_op_index;
        user_op_indices_by_supported_op[supported_op_index] = user_op_index;
    }

    return RoleMapping(std::move(supported_operand_indices_by_user_operand),
                       std::move(user_operand_indices_by_supported_operand), std::move(supported_op_indices_by_user_op),
                       std::move(user_op_indices_by_supported_op));
}

PatternOperandId RoleMapping::getSupportedOperandId(PatternOperandId user_operand_id) const {
    if (user_operand_id.getIndex() >= supported_operand_indices_by_user_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Operand role index %zu is out of range [0, %zu)",
                        user_operand_id.getIndex(), supported_operand_indices_by_user_operand_.size());
    }
    return PatternOperandId{supported_operand_indices_by_user_operand_[user_operand_id.getIndex()]};
}

PatternOperandId RoleMapping::getUserOperandId(PatternOperandId supported_operand_id) const {
    if (supported_operand_id.getIndex() >= user_operand_indices_by_supported_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Operand role index %zu is out of range [0, %zu)",
                        supported_operand_id.getIndex(), user_operand_indices_by_supported_operand_.size());
    }
    return PatternOperandId{user_operand_indices_by_supported_operand_[supported_operand_id.getIndex()]};
}

PatternOperationId RoleMapping::getSupportedOpId(PatternOperationId user_op_id) const {
    if (user_op_id.getIndex() >= supported_op_indices_by_user_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Op role index %zu is out of range [0, %zu)",
                        user_op_id.getIndex(), supported_op_indices_by_user_op_.size());
    }
    return PatternOperationId{supported_op_indices_by_user_op_[user_op_id.getIndex()]};
}

PatternOperationId RoleMapping::getUserOpId(PatternOperationId supported_op_id) const {
    if (supported_op_id.getIndex() >= user_op_indices_by_supported_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Op role index %zu is out of range [0, %zu)",
                        supported_op_id.getIndex(), user_op_indices_by_supported_op_.size());
    }
    return PatternOperationId{user_op_indices_by_supported_op_[supported_op_id.getIndex()]};
}

const OperationAttributes& OpArgument::getAttributes() const {
    if (!attributes_.has_value()) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Op argument has not been set"); }
    return *attributes_;
}

Args::Args(const Ops& ops, const Pattern& supported_pattern) : pattern_key_(ops.getPatternKey()) {
    const RoleMapping& role_mapping = ops.getRoleMapping();
    if (role_mapping.getNumOperands() != supported_pattern.getNumOperands() ||
        role_mapping.getNumOps() != supported_pattern.getNumOps()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "Supported PatternBuilder counts must match the Ops role mapping counts");
    }

    supported_operand_indices_by_user_operand_.reserve(role_mapping.getNumOperands());
    for (std::size_t user_operand_index = 0; user_operand_index < role_mapping.getNumOperands(); ++user_operand_index) {
        supported_operand_indices_by_user_operand_.push_back(
            role_mapping.getSupportedOperandId(PatternOperandId{user_operand_index}).getIndex());
    }
    supported_op_indices_by_user_op_.reserve(role_mapping.getNumOps());
    for (std::size_t user_op_index = 0; user_op_index < role_mapping.getNumOps(); ++user_op_index) {
        supported_op_indices_by_user_op_.push_back(
            role_mapping.getSupportedOpId(PatternOperationId{user_op_index}).getIndex());
    }

    operands_.reserve(supported_pattern.getNumOperands());
    for (std::size_t supported_operand_index = 0; supported_operand_index < supported_pattern.getNumOperands();
         ++supported_operand_index) {
        operands_.push_back(
            makeUnsetOperand(supported_pattern.getOperandNode(PatternOperandId{supported_operand_index}).getKind()));
    }

    op_arguments_.reserve(supported_pattern.getNumOps());
    for (std::size_t supported_op_index = 0; supported_op_index < supported_pattern.getNumOps(); ++supported_op_index) {
        op_arguments_.emplace_back(supported_pattern.getOpNode(PatternOperationId{supported_op_index}).getKind());
    }
}

bool Args::isComplete() const noexcept {
    return std::all_of(operands_.begin(), operands_.end(), isOperandSet) &&
           std::all_of(op_arguments_.begin(), op_arguments_.end(),
                       [](const OpArgument& argument) noexcept { return argument.isSet(); });
}

const OpsOperand& Args::getOperand(PatternOperandId supported_operand_id) const {
    if (supported_operand_id.getIndex() >= operands_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Operand role index %zu is out of range [0, %zu)",
                        supported_operand_id.getIndex(), operands_.size());
    }
    return operands_[supported_operand_id.getIndex()];
}

const OpArgument& Args::getOpArgument(PatternOperationId supported_op_id) const {
    if (supported_op_id.getIndex() >= op_arguments_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Op role index %zu is out of range [0, %zu)",
                        supported_op_id.getIndex(), op_arguments_.size());
    }
    return op_arguments_[supported_op_id.getIndex()];
}

std::size_t Args::mapUserOperand(PatternOperandId user_operand_id, OperandKind expected_kind) const {
    if (user_operand_id.getIndex() >= supported_operand_indices_by_user_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Operand role index %zu is out of range [0, %zu)",
                        user_operand_id.getIndex(), supported_operand_indices_by_user_operand_.size());
    }
    const std::size_t supported_operand_index = supported_operand_indices_by_user_operand_[user_operand_id.getIndex()];
    if (supported_operand_index >= operands_.size()) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Ops operand role mapping contains out-of-range slot %zu",
                        supported_operand_index);
    }
    if (getOperandKind(operands_[supported_operand_index]) != expected_kind) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Operand role %zu has a different storage family",
                        user_operand_id.getIndex());
    }
    return supported_operand_index;
}

std::size_t Args::mapUserOp(PatternOperationId user_op_id, OperationKind expected_kind) const {
    if (user_op_id.getIndex() >= supported_op_indices_by_user_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Op role index %zu is out of range [0, %zu)",
                        user_op_id.getIndex(), supported_op_indices_by_user_op_.size());
    }
    const std::size_t supported_op_index = supported_op_indices_by_user_op_[user_op_id.getIndex()];
    if (supported_op_index >= op_arguments_.size()) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Ops operation role mapping contains out-of-range slot %zu",
                        supported_op_index);
    }
    if (op_arguments_[supported_op_index].getKind() != expected_kind) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Op role %zu has a different operation kind",
                        user_op_id.getIndex());
    }
    return supported_op_index;
}

void Args::setOp(PatternOperationId user_op_id, OperationKind expected_kind, OperationAttributes&& attributes) {
    const std::size_t supported_op_index = mapUserOp(user_op_id, expected_kind);
    op_arguments_[supported_op_index].setAttributes(std::move(attributes));
}

}  // namespace ftrain
