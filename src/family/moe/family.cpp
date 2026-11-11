#include "flash_train/family/moe/family.hpp"

#include <utility>

namespace ftrain {

MoeProblem::MoeProblem(std::shared_ptr<MoeSession> session, MoeBindings bindings)
    : session_(std::move(session)), bindings_(std::move(bindings)) {
    if (session_ == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE attributes require a persistent session");
    }
    session_->validateBindings(bindings_);
}

std::vector<uint64_t> MoeProblem::getProblemKey() const {
    const auto& config = session_->getConfig();
    const auto& context = session_->getContext();
    std::vector<uint64_t> key{uint64_t(config.kind), uint64_t(context.device), uint64_t(context.rank),
                              uint64_t(context.world_size), uint64_t(config.m_dim), uint64_t(config.capacity_m),
                              uint64_t(config.n_dim), uint64_t(config.k_dim), uint64_t(config.num_experts),
                              uint64_t(config.topk), uint64_t(config.n_chunks),
                              uint64_t(bindings_.routing.storage.getNumericType()), uint64_t(config.route.size())};
    for (const unsigned char character : config.route) { key.push_back(character); }
    key.push_back(config.rank_m_sizes.size());
    for (const auto size : config.rank_m_sizes) { key.push_back(uint64_t(size)); }
    return key;
}

namespace {

class MoeRoutePrimitive final : public Primitive<MoeProblem> {
  public:
    explicit MoeRoutePrimitive(std::string name) : name_(std::move(name)) {}

    const char* getName() const noexcept override { return name_.c_str(); }

    std::unique_ptr<PrimitiveBase> clone() const override { return std::make_unique<MoeRoutePrimitive>(name_); }

    Result isApplicable(const MoeProblem& problem, const Constraints& constraints) const override {
        if (problem.getSession()->getConfig().route != name_) {
            return {FTRAIN_STATUS_UNSUPPORTED, "A different explicit MoE route was requested"};
        }
        if (problem.getSession()->getContext().device != constraints.getDeviceId()) {
            return {FTRAIN_STATUS_UNSUPPORTED, "MoE session belongs to a different HIP device"};
        }
        return {};
    }

    uint64_t getRequiredWorkspaceBytes() const noexcept override { return 0; }

    void configure(const MoeProblem& problem) override {
        session_ = problem.getSession();
        bindings_ = problem.getBindings();
    }

  private:
    void executeImpl(const Resources& resources) override {
        if (session_ == nullptr || !bindings_.has_value()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE Primitive has not been configured");
        }
        session_->execute(*bindings_, resources.getStream());
    }

    std::string name_;
    std::shared_ptr<MoeSession> session_;
    std::optional<MoeBindings> bindings_;
};

}

std::vector<std::shared_ptr<const Primitive<MoeProblem>>> makeMoeRecords(MoeKind kind) {
    const std::vector<std::string> routes = std::vector<std::string>{"ag_nn_ring_chunk", "ag_nn_a2a", "ag_nn_ring", "ag_nt_a2a", "ag_nt_a2a_chunk",
                                 "ag_nt_ring_chunk", "ag_nt_a2a_chunk_32k", "ag_nt_ring_chunk_32k"} ;
    std::vector<std::shared_ptr<const Primitive<MoeProblem>>> records;
    for (const auto& route : routes) { records.push_back(std::make_shared<MoeRoutePrimitive>(route)); }
    return records;
}

}
