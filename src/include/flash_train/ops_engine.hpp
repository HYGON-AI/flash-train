#ifndef FTRAIN_OPS_ENGINE_HPP_
#define FTRAIN_OPS_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "flash_train/matcher.hpp"
#include "flash_train/primitive.hpp"
#include "flash_train/runtime.hpp"

namespace ftrain {

// Returns the calling thread's current device. A platform runtime failure
// throws Exception with FTRAIN_STATUS_INTERNAL_ERROR.
FTrainDeviceId getCurrentDeviceId();

// One selection's runtime constraints: the device and the maximum workspace
// bytes a selected Primitive may require.
class SelectionContext final {
  public:
    SelectionContext(FTrainDeviceId device_id, std::uint64_t max_workspace_bytes) noexcept
        : device_id_(device_id), max_workspace_bytes_(max_workspace_bytes) {}

    FTrainDeviceId getDeviceId() const noexcept { return device_id_; }

    std::uint64_t getMaxWorkspaceBytes() const noexcept { return max_workspace_bytes_; }

  private:
    FTrainDeviceId device_id_;
    std::uint64_t max_workspace_bytes_;
};

// The cache key for one Primitive selection: device id, workspace byte limit,
// and the engine's argument tokens. Equality compares all three fields.
class SelectionKey final {
  public:
    SelectionKey(FTrainDeviceId device_id, std::uint64_t max_workspace_bytes,
                 std::vector<std::uint64_t>&& argument_tokens) noexcept;
    SelectionKey(const SelectionKey&)            = default;
    SelectionKey& operator=(const SelectionKey&) = default;
    SelectionKey(SelectionKey&& other) noexcept;
    SelectionKey& operator=(SelectionKey&& other) noexcept;

    bool operator==(const SelectionKey& other) const noexcept;

    bool operator!=(const SelectionKey& other) const noexcept { return !(*this == other); }

    std::size_t getHash() const noexcept { return hash_; }

  private:
    FTrainDeviceId device_id_;
    std::uint64_t max_workspace_bytes_;
    std::vector<std::uint64_t> argument_tokens_;
    std::size_t hash_;
};

class SelectionKeyHasher final {
  public:
    std::size_t operator()(const SelectionKey& key) const noexcept { return key.getHash(); }
};

using PrimitiveList = std::vector<std::shared_ptr<const PrimitiveBase>>;

// A candidate-selection policy consulted by OpsEngine, which visits Finders in
// registration order. Implementations must be safe for concurrent const calls.
class Finder {
  public:
    virtual ~Finder();

    // Returns whether this Finder participates for args and context; when
    // false, OpsEngine skips its other methods.
    virtual bool isEnabled(const Args& args, const SelectionContext& context) const = 0;

    // Returns the records to consider, usually a subset of records. Every
    // element must be non-null; a null candidate makes OpsEngine throw
    // Exception with FTRAIN_STATUS_INTERNAL_ERROR.
    virtual PrimitiveList findCandidates(const PrimitiveList& records, const Args& args,
                                         const SelectionContext& context) const = 0;

    // Reorders candidates from most to least preferred.
    virtual void sortCandidates(const Args& args, const SelectionContext& context, PrimitiveList& candidates) const = 0;
};

// A Primitive selection cache. Implementations must be safe for concurrent
// calls. publish() must keep the first record published for a key: a later
// publish for the same key must not replace it.
class PrimitiveCache {
  public:
    virtual ~PrimitiveCache();

    // Returns the record stored for key, or an empty pointer on a miss.
    virtual std::shared_ptr<const PrimitiveBase> find(const SelectionKey& key) const = 0;

    virtual void publish(const SelectionKey& key, std::shared_ptr<const PrimitiveBase> record) = 0;
};

class MemoryPrimitiveCache final : public PrimitiveCache {
  public:
    // Returns the record for an exact key, or an empty shared_ptr on a miss.
    std::shared_ptr<const PrimitiveBase> find(const SelectionKey& key) const override;

    // Publishes record only when key is absent. An existing mapping is retained.
    // A null record throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    void publish(const SelectionKey& key, std::shared_ptr<const PrimitiveBase> record) override;

    // Returns the number of stored records.
    std::size_t getSize() const;

  private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<SelectionKey, std::shared_ptr<const PrimitiveBase>, SelectionKeyHasher> records_;
};

// Applies the operational name lists: when enabled is non-empty, name must be
// in it, and name must not be in disabled.
inline bool isPrimitiveAllowed(const std::string& name, const std::unordered_set<std::string>& enabled,
                               const std::unordered_set<std::string>& disabled) {
    if (!enabled.empty() && enabled.find(name) == enabled.end()) { return false; }
    return disabled.find(name) == disabled.end();
}

// Returns the process-wide enabled Primitive names, parsed once on the first
// call from the comma-separated FTRAIN_ENABLED_PRIMITIVES environment
// variable; the variable is not re-read later. Names are trimmed of
// surrounding spaces and tabs.
const std::unordered_set<std::string>& getEnabledPrimitives();

// Returns the process-wide disabled Primitive names, parsed once on the first
// call from the comma-separated FTRAIN_DISABLED_PRIMITIVES environment
// variable; the variable is not re-read later. Names are trimmed of
// surrounding spaces and tabs.
const std::unordered_set<std::string>& getDisabledPrimitives();

// Base of every operator-family engine. Holds the supported Pattern's key,
// canonicalization, and schema. createPrimitive() is the selection entry
// point used by plans.
class OpsEngineBase {
  public:
    virtual ~OpsEngineBase();

    OpsEngineBase(const OpsEngineBase&)            = delete;
    OpsEngineBase& operator=(const OpsEngineBase&) = delete;
    OpsEngineBase(OpsEngineBase&&)                 = delete;
    OpsEngineBase& operator=(OpsEngineBase&&)      = delete;

    const PatternKey& getPatternKey() const noexcept { return supported_canonicalization_.getKey(); }

    const PatternCanonicalization& getSupportedCanonicalization() const noexcept { return supported_canonicalization_; }

    const SupportedSchema& getSupportedSchema() const noexcept { return supported_schema_; }

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
    PatternCanonicalization supported_canonicalization_;
    SupportedSchema supported_schema_;
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
        if (args.getPatternKey() != getPatternKey()) {
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

// Process-wide registry mapping PatternKeys to engines. Entries persist for
// the process lifetime and cannot be replaced or removed. All methods are
// thread-safe.
class Handle final {
  public:
    // Registers ops_engine under its PatternKey. A null engine or an
    // already-registered key throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT;
    // an existing entry is never replaced.
    void registerOpsEngine(std::shared_ptr<OpsEngineBase> ops_engine);

    // Returns the engine registered for pattern_key, or an empty shared_ptr.
    std::shared_ptr<const OpsEngineBase> findOpsEngine(const PatternKey& pattern_key) const;

    std::size_t getNumOpsEngines() const;

  private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<PatternKey, std::shared_ptr<const OpsEngineBase>, PatternKeyHasher> ops_engines_;
};

// Returns the process-wide Handle with every built-in engine already
// registered. Thread-safe; the first call performs the registration.
Handle& getGlobalHandle();

}  // namespace ftrain

#endif
