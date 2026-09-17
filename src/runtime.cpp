#include "flash_train/runtime.hpp"

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

RoleMapping RoleMapping::compose(const PatternCanonicalization& user_canonicalization,
                                 const PatternCanonicalization& supported_canonicalization) {
    if (user_canonicalization.getKey() != supported_canonicalization.getKey()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "Cannot compose role mappings whose canonical Pattern keys differ");
    }
    if (user_canonicalization.getNumOperands() != supported_canonicalization.getNumOperands() ||
        user_canonicalization.getNumOps() != supported_canonicalization.getNumOps()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Cannot compose role mappings with different role counts");
    }

    std::vector<std::size_t> supported_operand_indices_by_user_operand(user_canonicalization.getNumOperands(),
                                                                       kUnmappedRole);
    std::vector<std::size_t> user_operand_indices_by_supported_operand(supported_canonicalization.getNumOperands(),
                                                                       kUnmappedRole);
    for (std::size_t user_operand_index = 0; user_operand_index < user_canonicalization.getNumOperands();
         ++user_operand_index) {
        const OperandId user_operand_id{user_operand_index};
        const std::size_t canonical_operand_index = user_canonicalization.getCanonicalOperandIndex(user_operand_id);
        if (canonical_operand_index >= supported_canonicalization.getNumOperands() ||
            user_canonicalization.getOperandId(canonical_operand_index) != user_operand_id) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "User Operand canonical mapping is not a valid permutation at role %zu",
                            user_operand_index);
        }

        const OperandId supported_operand_id      = supported_canonicalization.getOperandId(canonical_operand_index);
        const std::size_t supported_operand_index = supported_operand_id.getIndex();
        if (supported_operand_index >= supported_canonicalization.getNumOperands() ||
            supported_canonicalization.getCanonicalOperandIndex(supported_operand_id) != canonical_operand_index ||
            user_operand_indices_by_supported_operand[supported_operand_index] != kUnmappedRole) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "Supported Operand canonical mapping is not a valid permutation at slot %zu",
                            canonical_operand_index);
        }
        supported_operand_indices_by_user_operand[user_operand_index]      = supported_operand_index;
        user_operand_indices_by_supported_operand[supported_operand_index] = user_operand_index;
    }

    std::vector<std::size_t> supported_op_indices_by_user_op(user_canonicalization.getNumOps(), kUnmappedRole);
    std::vector<std::size_t> user_op_indices_by_supported_op(supported_canonicalization.getNumOps(), kUnmappedRole);
    for (std::size_t user_op_index = 0; user_op_index < user_canonicalization.getNumOps(); ++user_op_index) {
        const OperationId user_op_id{user_op_index};
        const std::size_t canonical_op_index = user_canonicalization.getCanonicalOpIndex(user_op_id);
        if (canonical_op_index >= supported_canonicalization.getNumOps() ||
            user_canonicalization.getOperationId(canonical_op_index) != user_op_id) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "User Op canonical mapping is not a valid permutation at role %zu", user_op_index);
        }

        const OperationId supported_op_id    = supported_canonicalization.getOperationId(canonical_op_index);
        const std::size_t supported_op_index = supported_op_id.getIndex();
        if (supported_op_index >= supported_canonicalization.getNumOps() ||
            supported_canonicalization.getCanonicalOpIndex(supported_op_id) != canonical_op_index ||
            user_op_indices_by_supported_op[supported_op_index] != kUnmappedRole) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "Supported Op canonical mapping is not a valid permutation at slot %zu",
                            canonical_op_index);
        }
        supported_op_indices_by_user_op[user_op_index]      = supported_op_index;
        user_op_indices_by_supported_op[supported_op_index] = user_op_index;
    }

    return RoleMapping(std::move(supported_operand_indices_by_user_operand),
                       std::move(user_operand_indices_by_supported_operand), std::move(supported_op_indices_by_user_op),
                       std::move(user_op_indices_by_supported_op));
}

OperandId RoleMapping::getSupportedOperandId(OperandId user_operand_id) const {
    if (user_operand_id.getIndex() >= supported_operand_indices_by_user_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Operand role index %zu is out of range [0, %zu)",
                        user_operand_id.getIndex(), supported_operand_indices_by_user_operand_.size());
    }
    return OperandId{supported_operand_indices_by_user_operand_[user_operand_id.getIndex()]};
}

OperandId RoleMapping::getUserOperandId(OperandId supported_operand_id) const {
    if (supported_operand_id.getIndex() >= user_operand_indices_by_supported_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Operand role index %zu is out of range [0, %zu)",
                        supported_operand_id.getIndex(), user_operand_indices_by_supported_operand_.size());
    }
    return OperandId{user_operand_indices_by_supported_operand_[supported_operand_id.getIndex()]};
}

OperationId RoleMapping::getSupportedOpId(OperationId user_op_id) const {
    if (user_op_id.getIndex() >= supported_op_indices_by_user_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Op role index %zu is out of range [0, %zu)",
                        user_op_id.getIndex(), supported_op_indices_by_user_op_.size());
    }
    return OperationId{supported_op_indices_by_user_op_[user_op_id.getIndex()]};
}

OperationId RoleMapping::getUserOpId(OperationId supported_op_id) const {
    if (supported_op_id.getIndex() >= user_op_indices_by_supported_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Op role index %zu is out of range [0, %zu)",
                        supported_op_id.getIndex(), user_op_indices_by_supported_op_.size());
    }
    return OperationId{user_op_indices_by_supported_op_[supported_op_id.getIndex()]};
}

SupportedSchema::SupportedSchema(const Pattern& supported_pattern) {
    operand_kinds_.reserve(supported_pattern.getNumOperands());
    for (std::size_t operand_index = 0; operand_index < supported_pattern.getNumOperands(); ++operand_index) {
        operand_kinds_.push_back(supported_pattern.getOperandNode(OperandId{operand_index}).getKind());
    }

    op_kinds_.reserve(supported_pattern.getNumOps());
    for (std::size_t op_index = 0; op_index < supported_pattern.getNumOps(); ++op_index) {
        op_kinds_.push_back(supported_pattern.getOpNode(OperationId{op_index}).getKind());
    }
}

OperandKind SupportedSchema::getOperandKind(OperandId supported_operand_id) const {
    if (supported_operand_id.getIndex() >= operand_kinds_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Operand role index %zu is out of range [0, %zu)",
                        supported_operand_id.getIndex(), operand_kinds_.size());
    }
    return operand_kinds_[supported_operand_id.getIndex()];
}

OperationKind SupportedSchema::getOperationKind(OperationId supported_op_id) const {
    if (supported_op_id.getIndex() >= op_kinds_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Op role index %zu is out of range [0, %zu)",
                        supported_op_id.getIndex(), op_kinds_.size());
    }
    return op_kinds_[supported_op_id.getIndex()];
}

const OperationAttributes& OpArgument::getAttributes() const {
    if (!attributes_.has_value()) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Op argument has not been set"); }
    return *attributes_;
}

Args::Args(const Ops& ops, const SupportedSchema& supported_schema) : pattern_key_(ops.getPatternKey()) {
    const RoleMapping& role_mapping = ops.getRoleMapping();
    if (role_mapping.getNumOperands() != supported_schema.getNumOperands() ||
        role_mapping.getNumOps() != supported_schema.getNumOps()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "Supported schema counts must match the Ops role mapping counts");
    }

    supported_operand_indices_by_user_operand_.reserve(role_mapping.getNumOperands());
    for (std::size_t user_operand_index = 0; user_operand_index < role_mapping.getNumOperands(); ++user_operand_index) {
        supported_operand_indices_by_user_operand_.push_back(
            role_mapping.getSupportedOperandId(OperandId{user_operand_index}).getIndex());
    }
    supported_op_indices_by_user_op_.reserve(role_mapping.getNumOps());
    for (std::size_t user_op_index = 0; user_op_index < role_mapping.getNumOps(); ++user_op_index) {
        supported_op_indices_by_user_op_.push_back(
            role_mapping.getSupportedOpId(OperationId{user_op_index}).getIndex());
    }

    operands_.reserve(supported_schema.getNumOperands());
    for (std::size_t supported_operand_index = 0; supported_operand_index < supported_schema.getNumOperands();
         ++supported_operand_index) {
        operands_.push_back(makeUnsetOperand(supported_schema.getOperandKind(OperandId{supported_operand_index})));
    }

    op_arguments_.reserve(supported_schema.getNumOps());
    for (std::size_t supported_op_index = 0; supported_op_index < supported_schema.getNumOps(); ++supported_op_index) {
        op_arguments_.emplace_back(supported_schema.getOperationKind(OperationId{supported_op_index}));
    }
}

bool Args::isComplete() const noexcept {
    return std::all_of(operands_.begin(), operands_.end(), isOperandSet) &&
           std::all_of(op_arguments_.begin(), op_arguments_.end(),
                       [](const OpArgument& argument) noexcept { return argument.isSet(); });
}

const OpsOperand& Args::getOperand(OperandId supported_operand_id) const {
    if (supported_operand_id.getIndex() >= operands_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Operand role index %zu is out of range [0, %zu)",
                        supported_operand_id.getIndex(), operands_.size());
    }
    return operands_[supported_operand_id.getIndex()];
}

const OpArgument& Args::getOpArgument(OperationId supported_op_id) const {
    if (supported_op_id.getIndex() >= op_arguments_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Op role index %zu is out of range [0, %zu)",
                        supported_op_id.getIndex(), op_arguments_.size());
    }
    return op_arguments_[supported_op_id.getIndex()];
}

std::size_t Args::mapUserOperand(OperandId user_operand_id, OperandKind expected_kind) const {
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

std::size_t Args::mapUserOp(OperationId user_op_id, OperationKind expected_kind) const {
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

void Args::setTensor(OperandId user_operand_id, TensorStorage&& storage) {
    const std::size_t supported_operand_index = mapUserOperand(user_operand_id, OperandKind::kTensor);
    std::get<Tensor>(operands_[supported_operand_index]).setStorage(std::move(storage));
}

void Args::setTensorList(OperandId user_operand_id, TensorListStorage&& storage) {
    const std::size_t supported_operand_index = mapUserOperand(user_operand_id, OperandKind::kTensorList);
    std::get<TensorList>(operands_[supported_operand_index]).setStorage(std::move(storage));
}

void Args::setGroupedTensor(OperandId user_operand_id, GroupedTensorStorage&& storage) {
    const std::size_t supported_operand_index = mapUserOperand(user_operand_id, OperandKind::kGroupedTensor);
    std::get<GroupedTensor>(operands_[supported_operand_index]).setStorage(std::move(storage));
}

void Args::setOp(OperationId user_op_id, OperationKind expected_kind, OperationAttributes&& attributes) {
    const std::size_t supported_op_index = mapUserOp(user_op_id, expected_kind);
    op_arguments_[supported_op_index].setAttributes(std::move(attributes));
}

void Args::setGroupedABCDGemm(OperationId user_op_id, GroupedABCDGemmAttributes&& attributes) {
    setOp(user_op_id, OperationKind::kGroupedABCDGemm, OperationAttributes{std::move(attributes)});
}

void Args::setGemm(OperationId user_op_id, GemmAttributes&& attributes) {
    setOp(user_op_id, OperationKind::kGemm, OperationAttributes{std::move(attributes)});
}

void Args::setGroupedBCDGemm(OperationId user_op_id, GroupedBCDGemmAttributes&& attributes) {
    setOp(user_op_id, OperationKind::kGroupedBCDGemm, OperationAttributes{std::move(attributes)});
}

void Args::setGroupedABGemm(OperationId user_op_id, GroupedABGemmAttributes&& attributes) {
    setOp(user_op_id, OperationKind::kGroupedABGemm, OperationAttributes{std::move(attributes)});
}

}  // namespace ftrain
