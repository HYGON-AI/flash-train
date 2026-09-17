#ifndef FTRAIN_ENGINE_BASE_HPP_
#define FTRAIN_ENGINE_BASE_HPP_

#include <cstddef>
#include <memory>
#include <vector>

#include "flash_train/ops_args.hpp"
#include "flash_train/constraints.hpp"
#include "flash_train/cache.hpp"
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

    // Returns this call's Primitives, configured for args under constraints
    // and ordered best-first: every applicable registered record, each
    // enabled Finder's sorted candidates ahead of the records no Finder
    // offered, with primitive 0 the recommended default. Workspace budgets
    // are the caller's concern: compare each Primitive's
    // getRequiredWorkspaceBytes() against the budget and keep a fitting
    // index. Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT when
    // args' PatternKey differs from this engine's or args is incomplete;
    // the family's validation may additionally throw
    // FTRAIN_STATUS_INVALID_ARGUMENT or FTRAIN_STATUS_OVERFLOW for
    // inconsistent parameters, and FTRAIN_STATUS_UNSUPPORTED, naming every
    // rejection, when no registered Primitive applies. Successful
    // selections are cached per device and problem description as one
    // ordered record list; a cache hit skips the Finders and applicability
    // checks and reproduces the cached order. Safe for concurrent calls
    // with distinct Args objects.
    virtual std::vector<std::unique_ptr<PrimitiveBase>> createPrimitives(const Args& args,
                                                                         const Constraints& constraints) const = 0;

    // Returns one Plan for args on the calling thread's current device:
    // selects and configures the Plan's Primitives through createPrimitives()
    // and binds the Plan to that device. The Plan always holds at least one
    // Primitive, ordered best-first with primitive 0 the recommended
    // default. Throws exactly what createPrimitives() throws.
    Plan createPlan(const Args& args) const;

  protected:
    OpsEngineBase(const Pattern& supported_pattern);

  private:
    Pattern supported_pattern_;
};

// Engine for one operator family, assembled entirely at compile time; only
// the cache holds mutable runtime state. The engine draws everything
// family-specific from the Family policy's static functions and consults
// the Finder policies in pack order until one contributes an applicable
// candidate; records no Finder offered follow in registration order. A
// family joins by providing:
//
//   Family:
//     using Problem = ...               the family's parameter struct;
//                                       Problem must provide
//                                       getProblemKey(), whose comment
//                                       carries the cache-correctness
//                                       contract
//     static Pattern makePattern()
//     static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords()
//     static Problem makeProblem(const Args& args); the Problem type's
//                                       constructor validates the assembled
//                                       problem and rejects inconsistent
//                                       parameters
//
//   each Finder:
//     static const char* getName()          the name the comma-separated
//                                           FTRAIN_DISABLED_FINDERS variable
//                                           disables this Finder by
//     static bool isEnabled(const Problem& problem, const Constraints& constraints)
//     static std::vector<std::string> findCandidates(
//         const Problem& problem, const Constraints& constraints);
//                                           candidate names ranked
//                                           best-first; names matching no
//                                           registered record are skipped
template<typename Family, typename... Finders>
class OpsEngine : public OpsEngineBase {
  public:
    using Problem = typename Family::Problem;
    using Records = std::vector<std::shared_ptr<const Primitive<Problem>>>;

    // Assembles the engine from the Family policy: the supported Pattern,
    // the shared Primitive prototypes, and a fresh cache. A null record, a
    // null name, or a repeated name throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT; allocation failure throws
    // std::bad_alloc.
    OpsEngine() : OpsEngineBase(Family::makePattern()), records_(Family::makeRecords()) { indexRecordsByName(); }

    std::vector<std::unique_ptr<PrimitiveBase>> createPrimitives(const Args& args,
                                                                 const Constraints& constraints) const override {
        validateArgs(args);

        const Problem problem = Family::makeProblem(args);
        const CacheKey cache_key(constraints.getConstraintsKey(), problem.getProblemKey());

        if (!isSelectionCacheDisabled()) {
            const std::shared_ptr<const std::vector<std::string>> cached_names = cache_.find(cache_key);
            if (cached_names != nullptr) { return configureRecords(*cached_names, problem); }
        }
        return configureRecords(selectRecords(constraints, problem, cache_key), problem);
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

    // The selection proper: the applicable records' names in preference
    // order, published to the cache as one ordered list. Finders are
    // consulted in pack order until one contributes an applicable
    // candidate, so earlier Finders rank and later ones only cover the case
    // that every earlier Finder came up empty; records no Finder offered
    // follow in registration order, so a disabled or empty Finder degrades
    // ordering but never coverage. Throws Exception with
    // FTRAIN_STATUS_UNSUPPORTED, naming every rejection, when no record
    // applies.
    std::vector<std::string> selectRecords(const Constraints& constraints, const Problem& problem,
                                           const CacheKey& cache_key) const {
        std::vector<std::string> selected;
        std::unordered_set<const PrimitiveBase*> seen;
        std::string rejections;
        static_cast<void>((appendFinderRecords<Finders>(constraints, problem, selected, seen, rejections) || ...));
        for (const std::shared_ptr<const Primitive<Problem>>& record : records_) {
            appendIfApplicable(record, problem, constraints, selected, seen, rejections);
        }
        if (selected.empty()) {
            throw Exception(FTRAIN_STATUS_UNSUPPORTED,
                            "No Primitive supports the supplied Args and runtime constraints (%s)", rejections.c_str());
        }
        cache_.publish(cache_key, selected);
        return selected;
    }

    // Names identify records inside the engine, so every record must carry
    // a non-null, unique name; indexes the records by name once.
    void indexRecordsByName() {
        for (std::size_t index = 0; index < records_.size(); ++index) {
            const std::shared_ptr<const Primitive<Problem>>& record = records_[index];
            if (record == nullptr) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine Primitive %zu must not be null", index);
            }
            const char* name = record->getName();
            if (name == nullptr) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine Primitive %zu must not return a null name",
                                index);
            }
            if (!record_by_name_.emplace(std::string{name}, record).second) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine Primitive %zu repeats the name %s", index,
                                name);
            }
        }
    }

    // One Finder's contribution: its ranked candidate names, filtered to
    // the applicable ones and appended in offer order; names matching no
    // registered record are skipped. Returns whether any candidate was
    // appended. A Finder that is disabled by name or reports itself
    // disabled contributes nothing.
    template<typename Finder>
    bool appendFinderRecords(const Constraints& constraints, const Problem& problem, std::vector<std::string>& selected,
                             std::unordered_set<const PrimitiveBase*>& seen, std::string& rejections) const {
        if (isFinderDisabled(Finder::getName()) || !Finder::isEnabled(problem, constraints)) { return false; }

        bool appended = false;
        for (const std::string& candidate_name : Finder::findCandidates(problem, constraints)) {
            const auto record = record_by_name_.find(candidate_name);
            if (record == record_by_name_.end()) { continue; }
            appended = appendIfApplicable(record->second, problem, constraints, selected, seen, rejections) || appended;
        }
        return appended;
    }

    // Offers one record: already-seen records are skipped, records the
    // environment filters out by name are dropped silently, applicable
    // records contribute their names, and rejections are collected by
    // name. Returns whether the record was appended.
    bool appendIfApplicable(const std::shared_ptr<const Primitive<Problem>>& record, const Problem& problem,
                            const Constraints& constraints, std::vector<std::string>& selected,
                            std::unordered_set<const PrimitiveBase*>& seen, std::string& rejections) const {
        if (!seen.insert(record.get()).second) { return false; }
        if (!isPrimitiveAllowed(record->getName(), getEnabledPrimitives(), getDisabledPrimitives())) { return false; }
        const Result applicability = record->isApplicable(problem, constraints);
        if (!applicability.isSuccess()) {
            if (!rejections.empty()) { rejections += "; "; }
            rejections += record->getName();
            rejections += ": ";
            rejections += applicability.getMessage();
            return false;
        }
        selected.push_back(record->getName());
        return true;
    }

    std::unique_ptr<PrimitiveBase> createFromPrimitive(const PrimitiveBase& prototype, const Problem& problem) const {
        std::unique_ptr<PrimitiveBase> primitive = prototype.clone();
        if (primitive == nullptr) {
            throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Applicable Primitive returned a null clone");
        }
        static_cast<Primitive<Problem>&>(*primitive).configure(problem);
        return primitive;
    }

    std::vector<std::unique_ptr<PrimitiveBase>> configureRecords(const std::vector<std::string>& names,
                                                                 const Problem& problem) const {
        std::vector<std::unique_ptr<PrimitiveBase>> primitives;
        primitives.reserve(names.size());
        for (const std::string& name : names) {
            const auto record = record_by_name_.find(name);
            if (record == record_by_name_.end()) {
                throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Selection names unknown Primitive %s", name.c_str());
            }
            primitives.push_back(createFromPrimitive(*record->second, problem));
        }
        return primitives;
    }

    Records records_;
    std::unordered_map<std::string, std::shared_ptr<const Primitive<Problem>>> record_by_name_;
    // Selection is a const query; the cache is its memoization.
    mutable MemoryPrimitiveCache cache_;
};

}  // namespace ftrain

#endif
