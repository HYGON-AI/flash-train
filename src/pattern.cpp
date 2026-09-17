#include <cinttypes>
#include <type_traits>
#include <utility>

#include "flash_train/error.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {

OperandId Pattern::addOperand(OperandKind kind) {
    static_assert(std::is_nothrow_move_constructible_v<PatternOperandNode>);

    const std::size_t index = operand_nodes_.size();
    if (index == operand_nodes_.max_size()) {
        throw Exception(FTRAIN_STATUS_OVERFLOW, "Pattern operand capacity overflow");
    }

    operand_nodes_.emplace_back(kind);
    return OperandId{index};
}

const PatternOperandNode& Pattern::getOperandNode(OperandId operand_id) const {
    const std::uint64_t index = operand_id.getIndex();
    if (index >= operand_nodes_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "OperandId index " PRIu64 " is out of range for %zu operand nodes", index,
                        operand_nodes_.size());
    }
    return operand_nodes_[index];
}

OperationId Pattern::addOperation(OperationKind kind, std::vector<OperandId>&& inputs,
                                  std::vector<OperandId>&& outputs) {
    static_assert(std::is_nothrow_move_constructible_v<PatternOpNode>);
    static_assert(std::is_nothrow_constructible_v<OperationOutputPortId, OperationId, std::size_t>);

    const std::size_t op_index = op_nodes_.size();
    if (op_index == op_nodes_.max_size()) {
        throw Exception(FTRAIN_STATUS_OVERFLOW, "Pattern operation capacity overflow");
    }

    for (std::size_t port_index = 0; port_index < inputs.size(); ++port_index) {
        const std::uint64_t operand_index = inputs[port_index].getIndex();
        if (operand_index >= operand_nodes_.size()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "Input port %zu has out-of-range OperandId index " PRIu64 " for %zu operands", port_index,
                            operand_index, operand_nodes_.size());
        }

        const PatternOperandNode& operand_node = operand_nodes_[operand_index];
        for (std::size_t previous_port = 0; previous_port < port_index; ++previous_port) {
            if (inputs[previous_port] == inputs[port_index]) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Input ports %zu and %zu use the same OperandId index " PRIu64, previous_port,
                                port_index, operand_index);
            }
        }
        const auto& consumers = operand_node.getConsumers();
        if (consumers.size() == consumers.max_size()) {
            throw Exception(FTRAIN_STATUS_OVERFLOW, "Pattern operand consumer capacity overflow");
        }
    }

    for (std::size_t port_index = 0; port_index < outputs.size(); ++port_index) {
        const std::uint64_t operand_index = outputs[port_index].getIndex();
        if (operand_index >= operand_nodes_.size()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "Output port %zu has out-of-range OperandId index " PRIu64 " for %zu operands", port_index,
                            operand_index, operand_nodes_.size());
        }

        const PatternOperandNode& operand_node = operand_nodes_[operand_index];
        for (std::size_t input_port = 0; input_port < inputs.size(); ++input_port) {
            if (outputs[port_index] == inputs[input_port]) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Output port %zu reuses OperandId index " PRIu64 " from input port %zu", port_index,
                                operand_index, input_port);
            }
        }
        if (operand_node.getProducer().has_value()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OperandId index " PRIu64 " already has a producer",
                            operand_index);
        }
        for (std::size_t previous_port = 0; previous_port < port_index; ++previous_port) {
            if (outputs[previous_port] == outputs[port_index]) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Output ports %zu and %zu use the same OperandId index " PRIu64, previous_port,
                                port_index, operand_index);
            }
        }
    }

    const OperationId op_id{op_index};
    op_nodes_.emplace_back(kind, std::move(inputs), std::move(outputs));

    std::size_t num_added_consumers = 0;
    try {
        const auto& stored_inputs = op_nodes_.back().getInputs();
        for (std::size_t port_index = 0; port_index < stored_inputs.size(); ++port_index) {
            operand_nodes_[static_cast<std::size_t>(stored_inputs[port_index].getIndex())].consumers_.emplace_back(
                op_id, port_index);
            ++num_added_consumers;
        }
    }
    catch (...) {
        const auto& stored_inputs = op_nodes_.back().getInputs();
        while (num_added_consumers != 0) {
            --num_added_consumers;
            operand_nodes_[static_cast<std::size_t>(stored_inputs[num_added_consumers].getIndex())]
                .consumers_.pop_back();
        }
        op_nodes_.pop_back();
        throw;
    }

    const auto& stored_outputs = op_nodes_.back().getOutputs();
    for (std::size_t port_index = 0; port_index < stored_outputs.size(); ++port_index) {
        operand_nodes_[static_cast<std::size_t>(stored_outputs[port_index].getIndex())].producer_.emplace(op_id,
                                                                                                          port_index);
    }

    return op_id;
}

const PatternOpNode& Pattern::getOpNode(OperationId op_id) const {
    const std::uint64_t index = op_id.getIndex();
    if (index >= op_nodes_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "OperationId index " PRIu64 " is out of range for %zu operation nodes", index,
                        op_nodes_.size());
    }
    return op_nodes_[index];
}

}  // namespace ftrain
