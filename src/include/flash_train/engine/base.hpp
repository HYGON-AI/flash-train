#ifndef FTRAIN_ENGINE_BASE_HPP_
#define FTRAIN_ENGINE_BASE_HPP_

#include <cstddef>
#include <memory>
#include <vector>

#include "flash_train/ops_args.hpp"
#include "flash_train/context.hpp"
#include "flash_train/finder.hpp"
#include "flash_train/selection.hpp"
#include "flash_train/primitive/base.hpp"

namespace ftrain {

// Returns the calling thread's current device. A platform runtime failure
// throws Exception with FTRAIN_STATUS_INTERNAL_ERROR.
FTrainDeviceId getCurrentDeviceId();

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

    // Validates args, then returns a configured Primitive clone for it.
    // Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT when args'
    // PatternKey differs from this engine's or args is incomplete; the
    // family's validation may additionally throw FTRAIN_STATUS_INVALID_ARGUMENT
    // or FTRAIN_STATUS_OVERFLOW for inconsistent parameters. Throws Exception
    // with FTRAIN_STATUS_UNSUPPORTED, listing every candidate's rejection
    // reason, when no Primitive accepts the problem. Successful selections
    // are cached per device, workspace limit, and problem description; a
    // cache hit skips the Finder and applicability checks. Safe for
    // concurrent calls with distinct Args objects.
    virtual std::unique_ptr<PrimitiveBase> createPrimitive(const Args& args, const SelectionContext& context) const = 0;

  protected:
    OpsEngineBase(const Pattern& supported_pattern);

  private:
    Pattern supported_pattern_;
};

// Engine template for one operator family. The family's Problem struct,
// assembled by makeProblem() from positional Args, drives validation,
// selection, and configuration. Only the cache changes after construction.
template<typename Problem>
class OpsEngine : public OpsEngineBase {
  public:
    // Takes ownership of records and finders; both may be empty. Every record
    // must be a Primitive<Problem>. A null record or Finder throws
    // Exception with FTRAIN_STATUS_INVALID_ARGUMENT. Allocation failure
    // throws std::bad_alloc.
    OpsEngine(const Pattern& supported_pattern, PrimitiveList&& records,
              std::vector<std::shared_ptr<const Finder>>&& finders)
        : OpsEngineBase(supported_pattern), records_(std::move(records)), finders_(std::move(finders)),
          cache_(std::make_unique<MemoryPrimitiveCache>()) {
        validateDependencies();
    }

    // Same as the other constructor, but uses the supplied cache. A null cache
    // throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    OpsEngine(const Pattern& supported_pattern, PrimitiveList&& records,
              std::vector<std::shared_ptr<const Finder>>&& finders, std::unique_ptr<PrimitiveCache>&& cache)
        : OpsEngineBase(supported_pattern), records_(std::move(records)), finders_(std::move(finders)),
          cache_(std::move(cache)) {
        if (cache_ == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine cache must not be null"); }
        validateDependencies();
    }

    std::unique_ptr<PrimitiveBase> createPrimitive(const Args& args, const SelectionContext& context) const override {
        if (args.getPatternKey() != getPattern().getKey()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Args PatternKey does not match this OpsEngine");
        }
        if (!args.isComplete()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Args has one or more unset Operand or Op parameters");
        }

        const Problem problem = makeProblem(args);
        validateProblem(problem, context);
        const SelectionKey selection_key(context.getDeviceId(), context.getMaxWorkspaceBytes(),
                                         makeSelectionTokens(problem, context));

        const std::shared_ptr<const PrimitiveBase> cached_record = cache_->find(selection_key);
        if (cached_record != nullptr) { return createFromPrimitive(*cached_record, problem, context); }

        std::string rejections;
        for (const std::shared_ptr<const Finder>& finder : finders_) {
            if (!finder->isEnabled(args, context)) { continue; }

            PrimitiveList candidates = finder->findCandidates(records_, args, context);
            finder->sortCandidates(args, context, candidates);
            for (const std::shared_ptr<const PrimitiveBase>& candidate : candidates) {
                if (candidate == nullptr) {
                    throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Finder returned a null Primitive candidate");
                }
                if (!isPrimitiveAllowed(candidate->getName(), getEnabledPrimitives(), getDisabledPrimitives())) {
                    continue;
                }
                const Result applicability =
                    static_cast<const Primitive<Problem>&>(*candidate).isApplicable(problem, context);
                if (!applicability.isSuccess()) {
                    if (!rejections.empty()) { rejections += "; "; }
                    rejections += candidate->getName();
                    rejections += ": ";
                    rejections += applicability.getMessage();
                    continue;
                }

                std::unique_ptr<PrimitiveBase> primitive = createFromPrimitive(*candidate, problem, context);
                cache_->publish(selection_key, candidate);
                return primitive;
            }
        }

        throw Exception(FTRAIN_STATUS_UNSUPPORTED, "No Primitive supports the supplied Args and runtime context (%s)",
                        rejections.c_str());
    }

  protected:
    // Assembles the family's Problem from complete positional Args. Overrides
    // must be safe for concurrent const calls.
    virtual Problem makeProblem(const Args& args) const = 0;

    // Cross-field validation of problem; throw Exception to reject it. The
    // default accepts every problem.
    virtual void validateProblem(const Problem& problem, const SelectionContext& context) const {
        static_cast<void>(problem);
        static_cast<void>(context);
    }

    // Encodes every problem property that can change which Primitive applies
    // into integer tokens; problems with equal tokens must select the same
    // Primitive, because a cache hit skips the applicability check. Encode
    // memory addresses only through properties that affect applicability (for
    // example alignment or overlap), never as raw values. Variable-length
    // fields must include their element count so different encodings cannot
    // alias. Device and workspace limit are added by the engine.
    virtual std::vector<std::uint64_t> makeSelectionTokens(const Problem& problem,
                                                           const SelectionContext& context) const = 0;

  private:
    void validateDependencies() const {
        for (std::size_t index = 0; index < records_.size(); ++index) {
            if (records_[index] == nullptr) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine Primitive %zu must not be null", index);
            }
        }
        for (std::size_t index = 0; index < finders_.size(); ++index) {
            if (finders_[index] == nullptr) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine Finder %zu must not be null", index);
            }
        }
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

    PrimitiveList records_;
    std::vector<std::shared_ptr<const Finder>> finders_;
    std::unique_ptr<PrimitiveCache> cache_;
};

}  // namespace ftrain

#endif
