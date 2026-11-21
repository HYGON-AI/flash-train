#ifndef FTRAIN_FAMILY_MOE_FAMILY_HPP_
#define FTRAIN_FAMILY_MOE_FAMILY_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "flash_train/family/moe/session.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

class MoeProblem final {
  public:
    MoeProblem(std::shared_ptr<MoeSession> session, MoeBindings bindings);

    std::vector<uint64_t> getProblemKey() const;
    const std::shared_ptr<MoeSession>& getSession() const noexcept { return session_; }
    const MoeBindings& getBindings() const noexcept { return bindings_; }

  private:
    std::shared_ptr<MoeSession> session_;
    MoeBindings bindings_;
};

std::vector<std::shared_ptr<const Primitive<MoeProblem>>> makeMoeRecords(MoeKind kind);

template<OperationKind Kind>
struct MoeFamily final {
    using Problem = MoeProblem;
    using OpId = typename OperationTraits<Kind>::Id;
    static constexpr bool is_ag = Kind == OperationKind::kAGGroupGemm;
    static constexpr size_t num_ports = is_ag ? 4 : 5;

    struct Roles {
        Pattern pattern;
        std::array<FTrainTensorId, num_ports> ports;
        OpId operation;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder builder;
            std::array<FTrainTensorId, num_ports> ports;
            for (auto& port : ports) { port = builder.addOperand<OperandKind::kTensor>(); }
            OpId operation;
            if constexpr (is_ag) {
                operation = builder.addOperation<Kind>(ports[0], ports[1], ports[2], ports[3]);
            } else {
                operation = builder.addOperation<Kind>(ports[0], ports[1], ports[2], ports[3], ports[4]);
            }
            return Roles{builder.buildPattern(), ports, operation};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<MoeProblem>>> makeRecords() {
        return makeMoeRecords(is_ag ? MoeKind::kAG : MoeKind::kRS);
    }

    static MoeProblem makeProblem(const Args& args) {
        const auto& roles = makeRoles();
        const auto& attributes = std::get<MoeAttributes<Kind>>(
            *args.getOpArgument(OperationTraits<Kind>::createPatternOperationId(roles.operation)));
        auto tensor = [&](size_t index) {
            const auto& storage = std::get<Tensor>(*args.getOperand(
                OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ports[index]))).getStorageView();
            FTrainStorageView view{storage.getMemory(), storage.getDims().data(), nullptr,
                                   static_cast<uint8_t>(storage.getDims().size()), storage.getNumericType(),
                                   storage.getIndexType(), storage.isHostMemory()};
            if (view.index_type == FTRAIN_INDEX_TYPE_INVALID) { view.strides = storage.getStrides().data(); }
            const auto& metadata = attributes.tensors[index];
            return MoeTensor({view, metadata.identity, metadata.version});
        };
        MoeBindings bindings{tensor(0), tensor(1), tensor(2), tensor(num_ports - 1), std::nullopt};
        if constexpr (!is_ag) { bindings.expert_weight = tensor(3); }
        return MoeProblem(attributes.session, std::move(bindings));
    }
};

using AGGroupGemmFamily = MoeFamily<OperationKind::kAGGroupGemm>;
using MoEReduceRSFamily = MoeFamily<OperationKind::kMoEReduceRS>;

struct ExplicitMoeRouteFinder final {
    static const char* getName() { return "ExplicitMoeRoute"; }
    static std::vector<std::string> findCandidates(const MoeProblem& problem, const Constraints&) {
        return {problem.getSession()->getConfig().route};
    }
};

}

#endif
