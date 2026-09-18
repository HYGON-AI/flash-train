// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_ENGINE_HPP_
#define FTRAIN_ENGINE_HPP_

#include <cstddef>
#include <memory>
#include <vector>

#include "flash_train/ops_args.hpp"
#include "flash_train/constraints.hpp"
#include "flash_train/cache.hpp"
#include "flash_train/plan.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Base of every operator-family engine: holds the supported Pattern and
// its key. createPrimitives() is the selection entry point; createPlan()
// is the plan-facing wrapper.
class OpsEngineBase {
  public:
    virtual ~OpsEngineBase();

    OpsEngineBase(const OpsEngineBase&)            = delete;
    OpsEngineBase& operator=(const OpsEngineBase&) = delete;
    OpsEngineBase(OpsEngineBase&&)                 = delete;
    OpsEngineBase& operator=(OpsEngineBase&&)      = delete;

    const Pattern& getPattern() const noexcept { return supported_pattern_; }

    // Returns this call's Primitives, configured for args under constraints
    // and ordered best-first: the first Finder contributing an applicable
    // candidate supplies the whole ranked selection, and only when every
    // Finder comes up empty does the engine return every applicable record
    // in registration order; primitive 0 is the recommended default.
    // Workspace budgets are the caller's concern: compare each Primitive's
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
// candidate, whose ranked candidates become the selection; when none does,
// every applicable record follows in registration order. A family joins by
// providing:
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
            const std::shared_ptr<const std::vector<std::size_t>> cached_positions = cache_.find(cache_key);
            if (cached_positions != nullptr) { return configureRecords(*cached_positions, problem); }
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

    // The selection proper: the selected records' positions in preference
    // order, published to the cache as one ordered list. Finders are
    // consulted in pack order until one contributes an applicable
    // candidate; its ranked candidates are then the whole selection. Only
    // when every Finder came up empty does the engine walk the records in
    // registration order, and records a Finder already offered are never
    // re-checked. Throws Exception with FTRAIN_STATUS_UNSUPPORTED, naming
    // every rejection, when no record applies.
    std::vector<std::size_t> selectRecords(const Constraints& constraints, const Problem& problem,
                                           const CacheKey& cache_key) const {
        std::vector<std::size_t> selected;
        std::unordered_set<const PrimitiveBase*> seen;
        std::string rejections;
        const bool finder_selected =
            (appendFinderRecords<Finders>(constraints, problem, selected, seen, rejections) || ...);
        if (!finder_selected) {
            for (std::size_t index = 0; index < records_.size(); ++index) {
                appendIfApplicable(index, problem, constraints, selected, seen, rejections);
            }
        }
        if (selected.empty()) {
            throw Exception(FTRAIN_STATUS_UNSUPPORTED,
                            "No Primitive supports the supplied Args and runtime constraints (%s)", rejections.c_str());
        }
        cache_.publish(cache_key, selected);
        return selected;
    }

    // Names identify records inside the engine, so every record must carry
    // a non-null, unique name; indexes the records' positions by name once.
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
            if (!record_index_by_name_.emplace(std::string{name}, index).second) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OpsEngine Primitive %zu repeats the name %s", index,
                                name);
            }
        }
    }

    // One Finder's contribution: its ranked candidate names, filtered to
    // the applicable ones and appended in offer order; names matching no
    // registered record are skipped. Returns whether any candidate was
    // appended. A Finder disabled by name contributes nothing.
    template<typename Finder>
    bool appendFinderRecords(const Constraints& constraints, const Problem& problem, std::vector<std::size_t>& selected,
                             std::unordered_set<const PrimitiveBase*>& seen, std::string& rejections) const {
        if (isFinderDisabled(Finder::getName())) { return false; }

        bool appended = false;
        for (const std::string& candidate_name : Finder::findCandidates(problem, constraints)) {
            const auto index = record_index_by_name_.find(candidate_name);
            if (index == record_index_by_name_.end()) { continue; }
            appended = appendIfApplicable(index->second, problem, constraints, selected, seen, rejections) || appended;
        }
        return appended;
    }

    // Offers one record by its position in records_: already-seen records
    // are skipped, records the environment filters out by name are dropped
    // silently, applicable records contribute their positions, and
    // rejections are collected by name. Returns whether the record was
    // appended.
    bool appendIfApplicable(std::size_t index, const Problem& problem, const Constraints& constraints,
                            std::vector<std::size_t>& selected, std::unordered_set<const PrimitiveBase*>& seen,
                            std::string& rejections) const {
        const std::shared_ptr<const Primitive<Problem>>& record = records_[index];
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
        selected.push_back(index);
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

    // Clones and configures the records the positions name; positions come
    // from this engine's own selection walk, so every one is in range.
    std::vector<std::unique_ptr<PrimitiveBase>> configureRecords(const std::vector<std::size_t>& positions,
                                                                 const Problem& problem) const {
        std::vector<std::unique_ptr<PrimitiveBase>> primitives;
        primitives.reserve(positions.size());
        for (const std::size_t position : positions) {
            primitives.push_back(createFromPrimitive(*records_[position], problem));
        }
        return primitives;
    }

    // records_ owns the shared prototypes in registration order; the index
    // maps each record's name to its position in records_.
    Records records_;
    std::unordered_map<std::string, std::size_t> record_index_by_name_;
    // Selection is a const query; the cache is its memoization.
    mutable MemoryPrimitiveCache cache_;
};

// Returns every built-in engine. The process registry registers this
// whole list once; an engine family joins the library by constructing its
// engine in this list, and nowhere else. Allocation failure throws
// std::bad_alloc.
std::vector<std::shared_ptr<OpsEngineBase>> makeBuiltinOpsEngines();

}  // namespace ftrain

#endif
