#ifndef FTRAIN_ENGINE_BASE_HPP_
#define FTRAIN_ENGINE_BASE_HPP_

#include <cstddef>
#include <memory>
#include <vector>

#include "flash_train/ops_args.hpp"
#include "flash_train/context.hpp"
#include "flash_train/selection.hpp"
#include "flash_train/plan.hpp"
#include "flash_train/primitive/base.hpp"

namespace ftrain {

// Base of every operator-family engine. Holds the supported PatternBuilder's key,
// canonicalization, and schema. createPrimitive() is the selection entry
// point used by plans.
class OpsEngineBase {
  public:
    virtual ~OpsEngineBase();

    OpsEngineBase(const OpsEngineBase&)            = delete;
    OpsEngineBase& operator=(const OpsEngineBase&) = delete;
    OpsEngineBase(OpsEngineBase&&)                 = delete;
    OpsEngineBase& operator=(OpsEngineBase&&)      = delete;

    const Pattern& getPattern() const noexcept { return supported_pattern_; }

    // Returns this selection's Primitives, configured for args under
    // context and ordered best-first: exactly one Primitive when the cache
    // or a Finder selects it, or every applicable registered Primitive in
    // registration order when neither yields one -- the Finder-development
    // path. Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT when args'
    // PatternKey differs from this engine's or args is incomplete; the
    // family's validation may additionally throw
    // FTRAIN_STATUS_INVALID_ARGUMENT or FTRAIN_STATUS_OVERFLOW for
    // inconsistent parameters, and FTRAIN_STATUS_UNSUPPORTED, naming every
    // rejection, when no registered Primitive applies. Successful single
    // selections are cached per device, workspace limit, and problem
    // description; a cache hit skips the Finder and applicability checks.
    // Safe for concurrent calls with distinct Args objects.
    virtual std::vector<std::unique_ptr<PrimitiveBase>> createPrimitives(const Args& args,
                                                                         const SelectionContext& context) const = 0;

    // Returns one executable Plan for args under the calling thread's
    // current device and the max_workspace_bytes limit: selects and
    // configures the Plan's Primitives through createPrimitives() and binds
    // the Plan to that device; primitive 0 is the selection's default.
    // Throws exactly what createPrimitives() throws.
    Plan createPlan(const Args& args, std::uint64_t max_workspace_bytes) const;

  protected:
    OpsEngineBase(const Pattern& supported_pattern);

  private:
    Pattern supported_pattern_;
};

// Engine for one operator family, assembled entirely at compile time; only
// the cache holds mutable runtime state. The engine draws everything
// family-specific from the Family policy's static functions and consults
// the Finder policies in pack order until one yields an applicable
// candidate. A family joins by providing:
//
//   Family:
//     using Problem = ...               the family's parameter struct;
//                                       Problem must provide
//                                       getSelectionTokens(), whose comment
//                                       carries the cache-correctness
//                                       contract
//     static PatternBuilder makeSupportedPattern()
//     static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords()
//     static Problem makeProblem(const Args& args)
//     static void validateProblem(const Problem& problem, const SelectionContext& context)
//
//   each Finder:
//     static const char* getName()          the name the comma-separated
//                                           FTRAIN_DISABLED_FINDERS variable
//                                           disables this Finder by
//     static bool isEnabled(const Args& args, const SelectionContext& context)
//     static std::vector<std::shared_ptr<const Primitive<Problem>>> findCandidates(
//         const std::vector<std::shared_ptr<const Primitive<Problem>>>& records, const Args& args,
//         const SelectionContext& context)
//     static void sortCandidates(const Args& args, const SelectionContext& context,
//                                std::vector<std::shared_ptr<const Primitive<Problem>>>& candidates)
template<typename Family, typename... Finders>
class OpsEngine : public OpsEngineBase {
  public:
    using Problem = typename Family::Problem;
    using Records = std::vector<std::shared_ptr<const Primitive<Problem>>>;

    // Assembles the engine from the Family policy: the supported Pattern,
    // the shared Primitive prototypes, and a fresh cache. A null record
    // throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT; allocation
    // failure throws std::bad_alloc.
    OpsEngine()
        : OpsEngineBase(Family::makeSupportedPattern().buildPattern()), records_(Family::makeRecords()),
          cache_(std::make_unique<MemoryPrimitiveCache>()) {
        validateRecords();
    }

    // Same engine with an injected cache, for callers that observe cache
    // behavior. A null cache throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT.
    explicit OpsEngine(std::unique_ptr<PrimitiveCache>&& cache) : OpsEngine() {
        if (cache == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine cache must not be null"); }
        cache_ = std::move(cache);
    }

    std::vector<std::unique_ptr<PrimitiveBase>> createPrimitives(const Args& args,
                                                                 const SelectionContext& context) const override {
        validateArgs(args);

        const Problem problem = Family::makeProblem(args);
        Family::validateProblem(problem, context);
        const SelectionKey selection_key(context.getDeviceId(), context.getMaxWorkspaceBytes(),
                                         problem.getSelectionTokens());

        if (!isSelectionCacheDisabled()) {
            const std::shared_ptr<const PrimitiveBase> cached_record = cache_->find(selection_key);
            if (cached_record != nullptr) {
                std::vector<std::unique_ptr<PrimitiveBase>> primitives;
                primitives.push_back(createFromPrimitive(*cached_record, problem, context));
                return primitives;
            }
        }

        std::string rejections;
        std::unique_ptr<PrimitiveBase> selected;
        static_cast<void>((trySelectVia<Finders>(args, context, problem, selection_key, rejections, selected) || ...));
        if (selected != nullptr) {
            std::vector<std::unique_ptr<PrimitiveBase>> primitives;
            primitives.push_back(std::move(selected));
            return primitives;
        }

        return selectAllApplicable(context, problem);
    }

  private:
    void validateArgs(const Args& args) const {
        if (args.getPatternKey() != getPattern().getKey()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Args PatternKey does not match this OpsEngine");
        }
        if (!args.isComplete()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Args has one or more unset Operand or Op parameters");
        }
    }

    // Finder-development fallback: neither the cache nor any Finder yielded
    // a Primitive, so configure every applicable registered Primitive, in
    // registration order, and hand the whole set back; nothing is published
    // to the cache, because a finder-less enumeration is not a selection.
    // Throws Exception with FTRAIN_STATUS_UNSUPPORTED, naming every record's
    // rejection reason, when no record applies.
    std::vector<std::unique_ptr<PrimitiveBase>> selectAllApplicable(const SelectionContext& context,
                                                                    const Problem& problem) const {
        std::vector<std::unique_ptr<PrimitiveBase>> primitives;
        std::string rejections;
        for (const std::shared_ptr<const Primitive<Problem>>& record : records_) {
            if (!isPrimitiveAllowed(record->getName(), getEnabledPrimitives(), getDisabledPrimitives())) { continue; }
            const Result applicability = record->isApplicable(problem, context);
            if (!applicability.isSuccess()) {
                if (!rejections.empty()) { rejections += "; "; }
                rejections += record->getName();
                rejections += ": ";
                rejections += applicability.getMessage();
                continue;
            }
            primitives.push_back(createFromPrimitive(*record, problem, context));
        }
        if (primitives.empty()) {
            throw Exception(FTRAIN_STATUS_UNSUPPORTED,
                            "No Primitive supports the supplied Args and runtime context (%s)", rejections.c_str());
        }
        return primitives;
    }

    void validateRecords() const {
        for (std::size_t index = 0; index < records_.size(); ++index) {
            if (records_[index] == nullptr) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine Primitive %zu must not be null", index);
            }
        }
    }

    // One Finder's contribution to a selection: asks for candidates,
    // reorders them, and returns true with the first applicable candidate
    // configured and published once found, collecting rejections along the
    // way. A disabled Finder contributes nothing.
    template<typename Finder>
    bool trySelectVia(const Args& args, const SelectionContext& context, const Problem& problem,
                      const SelectionKey& selection_key, std::string& rejections,
                      std::unique_ptr<PrimitiveBase>& selected) const {
        if (isFinderDisabled(Finder::getName()) || !Finder::isEnabled(args, context)) { return false; }

        Records candidates = Finder::findCandidates(records_, args, context);
        Finder::sortCandidates(args, context, candidates);
        for (const std::shared_ptr<const Primitive<Problem>>& candidate : candidates) {
            if (candidate == nullptr) {
                throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Finder returned a null Primitive candidate");
            }
            if (!isPrimitiveAllowed(candidate->getName(), getEnabledPrimitives(), getDisabledPrimitives())) {
                continue;
            }
            const Result applicability = candidate->isApplicable(problem, context);
            if (!applicability.isSuccess()) {
                if (!rejections.empty()) { rejections += "; "; }
                rejections += candidate->getName();
                rejections += ": ";
                rejections += applicability.getMessage();
                continue;
            }

            selected = createFromPrimitive(*candidate, problem, context);
            cache_->publish(selection_key, candidate);
            return true;
        }
        return false;
    }

    std::unique_ptr<PrimitiveBase> createFromPrimitive(const PrimitiveBase& prototype, const Problem& problem,
                                                       const SelectionContext& context) const {
        std::unique_ptr<PrimitiveBase> primitive = prototype.clone();
        if (primitive == nullptr) {
            throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Applicable Primitive returned a null clone");
        }
        static_cast<Primitive<Problem>&>(*primitive).configure(problem);
        if (primitive->getRequiredWorkspaceBytes() > context.getMaxWorkspaceBytes()) {
            throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                            "Primitive requires %llu bytes above the %llu-byte workspace limit",
                            static_cast<unsigned long long>(primitive->getRequiredWorkspaceBytes()),
                            static_cast<unsigned long long>(context.getMaxWorkspaceBytes()));
        }
        return primitive;
    }

    Records records_;
    std::unique_ptr<PrimitiveCache> cache_;
};

}  // namespace ftrain

#endif
