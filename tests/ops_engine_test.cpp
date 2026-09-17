#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <thread>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/engine/base.hpp"
#include "flash_train/handle.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {
namespace {

template<typename Function>
void expectStatus(FTrainStatus status, Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "OpsEngine operation unexpectedly succeeded";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), status);
    }
}

struct RecordState {
    std::atomic<int> applicable_calls{0};
    std::atomic<int> create_calls{0};
};

enum class RecordCreation {
    kValid,
    kNull,
};

struct TestProblem {
    StorageView first;

    std::vector<std::uint64_t> getSelectionTokens() const;
};

class TestPrimitive final : public Primitive<TestProblem> {
  public:
    TestPrimitive(int record_id, bool applicable, std::uint64_t required_workspace_bytes,
                  RecordCreation creation = RecordCreation::kValid)
        : record_id_(record_id), applicable_(applicable), required_workspace_bytes_(required_workspace_bytes),
          creation_(creation), state_(std::make_shared<RecordState>()) {}

    const char* getName() const noexcept override { return "TestPrimitive"; }

    std::unique_ptr<PrimitiveBase> clone() const override {
        if (creation_ == RecordCreation::kNull) { return {}; }
        return std::make_unique<TestPrimitive>(*this);
    }

    int getRecordId() const noexcept { return record_id_; }

    void* getBoundMemory() const noexcept { return bound_memory_; }

    const std::shared_ptr<RecordState>& getState() const noexcept { return state_; }

    Result isApplicable(const TestProblem&, const SelectionContext&) const override {
        ++state_->applicable_calls;
        return applicable_ ? Result{} : Result(FTRAIN_STATUS_UNSUPPORTED, "test primitive is not applicable");
    }

    void configure(const TestProblem& problem) override {
        ++state_->create_calls;
        bound_memory_ = problem.first.getMemory();
    }

    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return required_workspace_bytes_; }

  private:
    void executeImpl(void*, std::uint64_t, FTrainStream) override {}

    int record_id_;
    void* bound_memory_ = nullptr;
    const bool applicable_;
    const std::uint64_t required_workspace_bytes_;
    const RecordCreation creation_;
    std::shared_ptr<RecordState> state_;
};

struct FinderState {
    std::atomic<int> enabled_calls{0};
    std::atomic<int> find_calls{0};
    std::atomic<int> sort_calls{0};
};

using TestRecords = std::vector<std::shared_ptr<const Primitive<TestProblem>>>;

// Finder personas: static policies whose per-scenario candidates and call
// counters live in per-type storage, so a test resets and arms the personas
// it uses before constructing its engine. OpsEngine consults them in pack
// order.
struct SingleFinder {
    static const char* getName() { return "Single"; }

    static inline TestRecords candidates;
    static inline std::shared_ptr<FinderState> state = std::make_shared<FinderState>();

    static bool isEnabled(const Args&, const SelectionContext&) {
        ++state->enabled_calls;
        return true;
    }

    static TestRecords findCandidates(const TestRecords&, const Args&, const SelectionContext&) {
        ++state->find_calls;
        return candidates;
    }

    static void sortCandidates(const Args&, const SelectionContext&, TestRecords&) { ++state->sort_calls; }
};

struct DisabledFinder {
    static const char* getName() { return "Disabled"; }

    static inline TestRecords candidates;
    static inline std::shared_ptr<FinderState> state = std::make_shared<FinderState>();

    static bool isEnabled(const Args&, const SelectionContext&) {
        ++state->enabled_calls;
        return false;
    }

    static TestRecords findCandidates(const TestRecords&, const Args&, const SelectionContext&) {
        ++state->find_calls;
        return candidates;
    }

    static void sortCandidates(const Args&, const SelectionContext&, TestRecords&) { ++state->sort_calls; }
};

struct EmptyFinder {
    static const char* getName() { return "Empty"; }

    static inline TestRecords candidates;
    static inline std::shared_ptr<FinderState> state = std::make_shared<FinderState>();

    static bool isEnabled(const Args&, const SelectionContext&) {
        ++state->enabled_calls;
        return true;
    }

    static TestRecords findCandidates(const TestRecords&, const Args&, const SelectionContext&) {
        ++state->find_calls;
        return {};
    }

    static void sortCandidates(const Args&, const SelectionContext&, TestRecords&) { ++state->sort_calls; }
};

struct ReversingFinder {
    static const char* getName() { return "Reversing"; }

    static inline TestRecords candidates;
    static inline std::shared_ptr<FinderState> state = std::make_shared<FinderState>();

    static bool isEnabled(const Args&, const SelectionContext&) {
        ++state->enabled_calls;
        return true;
    }

    static TestRecords findCandidates(const TestRecords&, const Args&, const SelectionContext&) {
        ++state->find_calls;
        return candidates;
    }

    static void sortCandidates(const Args&, const SelectionContext&, TestRecords& sorted_candidates) {
        ++state->sort_calls;
        std::reverse(sorted_candidates.begin(), sorted_candidates.end());
    }
};

struct UnreachableFinder {
    static const char* getName() { return "Unreachable"; }

    static inline TestRecords candidates;
    static inline std::shared_ptr<FinderState> state = std::make_shared<FinderState>();

    static bool isEnabled(const Args&, const SelectionContext&) {
        ++state->enabled_calls;
        return true;
    }

    static TestRecords findCandidates(const TestRecords&, const Args&, const SelectionContext&) {
        ++state->find_calls;
        return candidates;
    }

    static void sortCandidates(const Args&, const SelectionContext&, TestRecords&) { ++state->sort_calls; }
};

template<typename Persona>
void armPersona(TestRecords candidates) {
    Persona::candidates = std::move(candidates);
    Persona::state      = std::make_shared<FinderState>();
}

enum class TestSelectionTokenTag : std::uint64_t {
    kNumericType,
    kIndexType,
    kDims,
    kStrides,
    kHostMemory,
    kHasMemory,
};

// The test family policy: one Tensor operand; the scenario's records come
// from per-test static storage armed before engine construction.
struct TestFamily {
    using Problem = TestProblem;

    static inline TestRecords records;

    static PatternBuilder makeSupportedPattern();

    static TestRecords makeRecords() { return records; }

    static TestProblem makeProblem(const Args& args) {
        return TestProblem{std::get<Tensor>(*args.getOperand(PatternOperandId{0})).getStorageView()};
    }

    static void validateProblem(const TestProblem&, const SelectionContext&) {}
};

std::vector<std::uint64_t> TestProblem::getSelectionTokens() const {
    const StorageView& view = first;

    std::vector<std::uint64_t> tokens;
    tokens.reserve(8 + view.getDims().size() + view.getStrides().size());
    tokens.push_back(static_cast<std::uint64_t>(TestSelectionTokenTag::kNumericType));
    tokens.push_back(static_cast<std::uint64_t>(view.getNumericType()));
    tokens.push_back(static_cast<std::uint64_t>(TestSelectionTokenTag::kIndexType));
    tokens.push_back(static_cast<std::uint64_t>(view.getIndexType()));
    tokens.push_back(static_cast<std::uint64_t>(TestSelectionTokenTag::kDims));
    tokens.push_back(static_cast<std::uint64_t>(view.getDims().size()));
    for (const std::int64_t dim : view.getDims()) { tokens.push_back(static_cast<std::uint64_t>(dim)); }
    tokens.push_back(static_cast<std::uint64_t>(TestSelectionTokenTag::kStrides));
    tokens.push_back(static_cast<std::uint64_t>(view.getStrides().size()));
    for (const std::int64_t stride : view.getStrides()) { tokens.push_back(static_cast<std::uint64_t>(stride)); }
    tokens.push_back(static_cast<std::uint64_t>(TestSelectionTokenTag::kHostMemory));
    tokens.push_back(static_cast<std::uint64_t>(view.isHostMemory()));
    tokens.push_back(static_cast<std::uint64_t>(TestSelectionTokenTag::kHasMemory));
    tokens.push_back(static_cast<std::uint64_t>(view.getMemory() != nullptr));
    return tokens;
}

PatternBuilder makeTensorPattern() {
    PatternBuilder pattern;
    static_cast<void>(pattern.addOperand<OperandKind::kTensor>());
    return pattern;
}

PatternBuilder TestFamily::makeSupportedPattern() { return makeTensorPattern(); }

Ops makeOps(const PatternBuilder& user_pattern, const OpsEngineBase& engine) {
    return Ops(user_pattern.buildPattern(), engine.getPattern());
}

Args makeArgs(const PatternBuilder& user_pattern, const OpsEngineBase& engine, std::vector<std::int64_t> dims,
              FTrainNumericType numeric_type = FTRAIN_NUMERIC_TYPE_FP32, void* memory = nullptr) {
    Ops ops   = makeOps(user_pattern, engine);
    Args args = ops.makeArgs();

    static std::uint32_t default_memory = 0;
    FTrainStorageView view{};
    view.memory         = memory == nullptr ? &default_memory : memory;
    view.dims           = dims.empty() ? nullptr : dims.data();
    view.strides        = nullptr;
    view.num_dims       = static_cast<std::uint8_t>(dims.size());
    view.numeric_type   = numeric_type;
    view.index_type     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    view.is_host_memory = false;
    args.setOperand<OperandKind::kTensor>(PatternOperandId{0}, Tensor{StorageView{view}});
    return args;
}

// Runs one selection and returns its single Primitive; every normal-path
// test expects exactly one.
template<typename Engine>
std::unique_ptr<PrimitiveBase> selectOne(const Engine& engine, const Args& args, const SelectionContext& context) {
    std::vector<std::unique_ptr<PrimitiveBase>> primitives = engine.createPrimitives(args, context);
    EXPECT_EQ(primitives.size(), 1);
    return std::move(primitives[0]);
}

int getRecordId(const std::unique_ptr<PrimitiveBase>& primitive) {
    return dynamic_cast<const TestPrimitive&>(*primitive).getRecordId();
}

void* getBoundMemory(const std::unique_ptr<PrimitiveBase>& primitive) {
    return dynamic_cast<const TestPrimitive&>(*primitive).getBoundMemory();
}

static_assert(std::has_virtual_destructor_v<PrimitiveBase>);
static_assert(std::has_virtual_destructor_v<PrimitiveCache>);
static_assert(!std::is_copy_constructible_v<OpsEngineBase>);
static_assert(!std::is_move_constructible_v<OpsEngineBase>);
static_assert(noexcept(std::declval<const SelectionContext&>().getDeviceId()));
static_assert(noexcept(std::declval<const SelectionContext&>().getMaxWorkspaceBytes()));
static_assert(noexcept(std::declval<const SelectionKey&>().getHash()));
static_assert(std::is_nothrow_move_constructible_v<SelectionKey>);
static_assert(std::is_nothrow_move_assignable_v<SelectionKey>);

TEST(SelectionKeyTest, UsesDeviceWorkspaceAndAllArgumentTokensForExactEquality) {
    const SelectionKey key{
        2, 4096, std::vector<std::uint64_t>{3, 5, 8}
    };
    const SelectionKey same_key{
        2, 4096, std::vector<std::uint64_t>{3, 5, 8}
    };
    const SelectionKey different_device{
        3, 4096, std::vector<std::uint64_t>{3, 5, 8}
    };
    const SelectionKey different_workspace{
        2, 8192, std::vector<std::uint64_t>{3, 5, 8}
    };
    const SelectionKey different_tokens{
        2, 4096, std::vector<std::uint64_t>{3, 5, 9}
    };

    EXPECT_EQ(key, same_key);
    EXPECT_EQ(key.getHash(), same_key.getHash());
    EXPECT_NE(key, different_device);
    EXPECT_NE(key, different_workspace);
    EXPECT_NE(key, different_tokens);
}

TEST(SelectionKeyTest, PreservesHashInvariantAcrossMoves) {
    SelectionKey first_key{
        2, 4096, std::vector<std::uint64_t>{3, 5, 8}
    };
    SelectionKey second_key{
        2, 4096, std::vector<std::uint64_t>{13, 21}
    };
    const SelectionKey expected_first_key  = first_key;
    const SelectionKey expected_second_key = second_key;

    SelectionKey moved_first_key(std::move(first_key));
    SelectionKey moved_second_key = expected_first_key;
    moved_second_key              = std::move(second_key);

    EXPECT_EQ(moved_first_key, expected_first_key);
    EXPECT_EQ(moved_second_key, expected_second_key);
    if (first_key == second_key) { EXPECT_EQ(first_key.getHash(), second_key.getHash()); }
}

TEST(MemoryPrimitiveCacheTest, PublishesOnceAndNeverReplacesAnExactKey) {
    MemoryPrimitiveCache cache;
    const SelectionKey key{0, 1024, std::vector<std::uint64_t>{7}};
    const auto first_record  = std::make_shared<TestPrimitive>(1, true, 0);
    const auto second_record = std::make_shared<TestPrimitive>(2, true, 0);

    EXPECT_EQ(cache.find(key), nullptr);
    cache.publish(key, first_record);
    cache.publish(key, second_record);

    EXPECT_EQ(cache.getSize(), 1);
    EXPECT_EQ(cache.find(key), first_record);
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] { cache.publish(key, nullptr); });
}

TEST(MemoryPrimitiveCacheTest, SupportsConcurrentPublicationAndLookup) {
    MemoryPrimitiveCache cache;
    constexpr std::size_t kNumThreads = 8;
    std::vector<std::shared_ptr<TestPrimitive>> records;
    std::vector<std::thread> threads;
    records.reserve(kNumThreads);
    threads.reserve(kNumThreads);

    for (std::size_t index = 0; index < kNumThreads; ++index) {
        records.push_back(std::make_shared<TestPrimitive>(static_cast<int>(index), true, 0));
        threads.emplace_back([&, index] {
            const SelectionKey key{0, 1024, std::vector<std::uint64_t>{index}};
            cache.publish(key, records[index]);
            EXPECT_EQ(cache.find(key), records[index]);
        });
    }
    for (std::thread& thread : threads) { thread.join(); }

    EXPECT_EQ(cache.getSize(), kNumThreads);
}

TEST(MemoryPrimitiveCacheTest, ConcurrentExactKeyPublicationKeepsOneRecordWithoutReplacement) {
    MemoryPrimitiveCache cache;
    constexpr std::size_t kNumThreads = 8;
    const SelectionKey key{
        0, 1024, std::vector<std::uint64_t>{13, 21}
    };
    std::vector<std::shared_ptr<TestPrimitive>> records;
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(kNumThreads);
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    records.reserve(kNumThreads);
    threads.reserve(kNumThreads);
    const auto initial_record = std::make_shared<TestPrimitive>(-1, true, 0);
    cache.publish(key, initial_record);

    for (std::size_t index = 0; index < kNumThreads; ++index) {
        records.push_back(std::make_shared<TestPrimitive>(static_cast<int>(index), true, 0));
        threads.emplace_back([&, index] {
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            try {
                cache.publish(key, records[index]);
            }
            catch (...) {
                errors[index] = std::current_exception();
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kNumThreads) { std::this_thread::yield(); }
    start.store(true, std::memory_order_release);
    for (std::thread& thread : threads) { thread.join(); }

    for (const std::exception_ptr& error : errors) { EXPECT_FALSE(error); }
    const std::shared_ptr<const PrimitiveBase> selected = cache.find(key);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(cache.getSize(), 1);
    EXPECT_EQ(selected, initial_record);
}

TEST(OpsEngineTest, CachesAnExactSelectionAndBindsANewPrimitiveOnEveryCall) {
    const PatternBuilder pattern = makeTensorPattern();
    const auto record            = std::make_shared<TestPrimitive>(17, true, 32);
    TestFamily::records          = TestRecords{record};
    armPersona<SingleFinder>(TestRecords{record});
    auto cache                           = std::make_unique<MemoryPrimitiveCache>();
    MemoryPrimitiveCache* cache_observer = cache.get();
    OpsEngine<TestFamily, SingleFinder> engine(std::move(cache));
    const Args args = makeArgs(pattern, engine, {4, 8});
    const SelectionContext context{getCurrentDeviceId(), 64};

    const std::unique_ptr<PrimitiveBase> first  = selectOne(engine, args, context);
    const std::unique_ptr<PrimitiveBase> second = selectOne(engine, args, context);

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_EQ(getRecordId(first), 17);
    EXPECT_EQ(getRecordId(second), 17);
    EXPECT_EQ(record->getState()->applicable_calls.load(), 1);
    EXPECT_EQ(record->getState()->create_calls.load(), 2);
    EXPECT_EQ(SingleFinder::state->find_calls.load(), 1);
    EXPECT_EQ(SingleFinder::state->sort_calls.load(), 1);
    EXPECT_EQ(cache_observer->getSize(), 1);
}

TEST(OpsEngineTest, ReusesACompatibleRecordButBindsEachArgsSnapshotIndependently) {
    const PatternBuilder pattern = makeTensorPattern();
    const auto record            = std::make_shared<TestPrimitive>(19, true, 0);
    TestFamily::records          = TestRecords{record};
    armPersona<SingleFinder>(TestRecords{record});
    OpsEngine<TestFamily, SingleFinder> engine;
    const SelectionContext context{getCurrentDeviceId(), 0};
    std::uint32_t first_memory  = 0;
    std::uint32_t second_memory = 0;
    std::unique_ptr<PrimitiveBase> first;
    std::unique_ptr<PrimitiveBase> second;

    {
        const Args first_args = makeArgs(pattern, engine, {64}, FTRAIN_NUMERIC_TYPE_FP32, &first_memory);
        first                 = selectOne(engine, first_args, context);
    }
    {
        const Args second_args = makeArgs(pattern, engine, {64}, FTRAIN_NUMERIC_TYPE_FP32, &second_memory);
        second                 = selectOne(engine, second_args, context);
    }

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(getBoundMemory(first), &first_memory);
    EXPECT_EQ(getBoundMemory(second), &second_memory);
    EXPECT_EQ(SingleFinder::state->find_calls.load(), 1);
    EXPECT_EQ(record->getState()->applicable_calls.load(), 1);
    EXPECT_EQ(record->getState()->create_calls.load(), 2);
    EXPECT_NO_THROW(first->execute(nullptr, 0, nullptr));
    EXPECT_NO_THROW(second->execute(nullptr, 0, nullptr));
}

TEST(OpsEngineTest, ConcurrentExactKeyCreationPublishesOnceAndThenStablyHitsCache) {
    constexpr std::size_t kNumThreads = 8;
    const PatternBuilder pattern      = makeTensorPattern();
    const auto record                 = std::make_shared<TestPrimitive>(71, true, 0);
    TestFamily::records               = TestRecords{record};
    armPersona<SingleFinder>(TestRecords{record});
    auto cache                           = std::make_unique<MemoryPrimitiveCache>();
    MemoryPrimitiveCache* cache_observer = cache.get();
    OpsEngine<TestFamily, SingleFinder> engine(std::move(cache));
    const SelectionContext context{getCurrentDeviceId(), 0};
    std::vector<Args> arguments;
    std::vector<std::unique_ptr<PrimitiveBase>> primitives(kNumThreads);
    std::vector<std::exception_ptr> errors(kNumThreads);
    std::vector<std::thread> threads;
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    arguments.reserve(kNumThreads);
    threads.reserve(kNumThreads);
    for (std::size_t index = 0; index < kNumThreads; ++index) { arguments.push_back(makeArgs(pattern, engine, {64})); }

    for (std::size_t index = 0; index < kNumThreads; ++index) {
        threads.emplace_back([&, index] {
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            try {
                primitives[index] = selectOne(engine, arguments[index], context);
            }
            catch (...) {
                errors[index] = std::current_exception();
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kNumThreads) { std::this_thread::yield(); }
    start.store(true, std::memory_order_release);
    for (std::thread& thread : threads) { thread.join(); }

    for (std::size_t index = 0; index < kNumThreads; ++index) {
        EXPECT_FALSE(errors[index]);
        ASSERT_NE(primitives[index], nullptr);
        EXPECT_EQ(getRecordId(primitives[index]), 71);
    }
    const int finder_calls_after_race     = SingleFinder::state->find_calls.load();
    const int applicable_calls_after_race = record->getState()->applicable_calls.load();
    EXPECT_GE(finder_calls_after_race, 1);
    EXPECT_LE(finder_calls_after_race, static_cast<int>(kNumThreads));
    EXPECT_EQ(applicable_calls_after_race, finder_calls_after_race);
    EXPECT_EQ(record->getState()->create_calls.load(), static_cast<int>(kNumThreads));
    EXPECT_EQ(cache_observer->getSize(), 1);

    const std::unique_ptr<PrimitiveBase> cached = selectOne(engine, arguments.front(), context);
    ASSERT_NE(cached, nullptr);
    EXPECT_EQ(getRecordId(cached), 71);
    EXPECT_EQ(SingleFinder::state->find_calls.load(), finder_calls_after_race);
    EXPECT_EQ(record->getState()->applicable_calls.load(), applicable_calls_after_race);
    EXPECT_EQ(record->getState()->create_calls.load(), static_cast<int>(kNumThreads + 1));
}

TEST(OpsEngineTest, SeparatesCacheEntriesByArgumentsDeviceAndWorkspaceLimit) {
    const PatternBuilder pattern = makeTensorPattern();
    const auto record            = std::make_shared<TestPrimitive>(23, true, 0);
    TestFamily::records          = TestRecords{record};
    armPersona<SingleFinder>(TestRecords{record});
    auto cache                           = std::make_unique<MemoryPrimitiveCache>();
    MemoryPrimitiveCache* cache_observer = cache.get();
    OpsEngine<TestFamily, SingleFinder> engine(std::move(cache));
    const Args first_args          = makeArgs(pattern, engine, {16});
    const Args second_args         = makeArgs(pattern, engine, {32});
    const FTrainDeviceId device_id = getCurrentDeviceId();

    static_cast<void>(engine.createPrimitives(first_args, SelectionContext{device_id, 64}));
    static_cast<void>(engine.createPrimitives(second_args, SelectionContext{device_id, 64}));
    static_cast<void>(engine.createPrimitives(first_args, SelectionContext{device_id, 128}));
    static_cast<void>(
        engine.createPrimitives(first_args, SelectionContext{static_cast<FTrainDeviceId>(device_id + 1), 64}));

    EXPECT_EQ(SingleFinder::state->find_calls.load(), 4);
    EXPECT_EQ(record->getState()->applicable_calls.load(), 4);
    EXPECT_EQ(cache_observer->getSize(), 4);
}

TEST(OpsEngineTest, VisitsFindersInOrderAndSelectsFirstApplicableSortedCandidate) {
    const PatternBuilder pattern = makeTensorPattern();
    const auto inapplicable      = std::make_shared<TestPrimitive>(31, false, 0);
    const auto selected          = std::make_shared<TestPrimitive>(37, true, 0);
    const auto later             = std::make_shared<TestPrimitive>(41, true, 0);
    TestFamily::records          = TestRecords{inapplicable, selected, later};
    armPersona<DisabledFinder>(TestRecords{later});
    armPersona<EmptyFinder>({});
    armPersona<ReversingFinder>(TestRecords{selected, inapplicable});
    armPersona<UnreachableFinder>(TestRecords{later});
    OpsEngine<TestFamily, DisabledFinder, EmptyFinder, ReversingFinder, UnreachableFinder> engine;
    const Args args = makeArgs(pattern, engine, {8});

    const std::unique_ptr<PrimitiveBase> primitive = selectOne(engine, args, SelectionContext{getCurrentDeviceId(), 0});

    ASSERT_NE(primitive, nullptr);
    EXPECT_EQ(getRecordId(primitive), 37);
    EXPECT_EQ(DisabledFinder::state->find_calls.load(), 0);
    EXPECT_EQ(EmptyFinder::state->find_calls.load(), 1);
    EXPECT_EQ(ReversingFinder::state->find_calls.load(), 1);
    EXPECT_EQ(inapplicable->getState()->applicable_calls.load(), 1);
    EXPECT_EQ(selected->getState()->applicable_calls.load(), 1);
    EXPECT_EQ(UnreachableFinder::state->enabled_calls.load(), 0);
    EXPECT_EQ(later->getState()->applicable_calls.load(), 0);
}

TEST(OpsEngineTest, EnumeratesEveryApplicablePrimitiveWhenFindersYieldNothing) {
    const PatternBuilder pattern = makeTensorPattern();
    const auto rejected          = std::make_shared<TestPrimitive>(83, false, 0);
    const auto first_applicable  = std::make_shared<TestPrimitive>(89, true, 0);
    const auto second_applicable = std::make_shared<TestPrimitive>(97, true, 0);
    TestFamily::records          = TestRecords{rejected, first_applicable, second_applicable};
    armPersona<SingleFinder>(TestRecords{rejected});
    auto cache                           = std::make_unique<MemoryPrimitiveCache>();
    MemoryPrimitiveCache* cache_observer = cache.get();
    OpsEngine<TestFamily, SingleFinder> engine(std::move(cache));
    const Args args = makeArgs(pattern, engine, {8});

    const std::vector<std::unique_ptr<PrimitiveBase>> primitives =
        engine.createPrimitives(args, SelectionContext{getCurrentDeviceId(), 64});

    ASSERT_EQ(primitives.size(), 2);
    EXPECT_EQ(getRecordId(primitives[0]), 89);
    EXPECT_EQ(getRecordId(primitives[1]), 97);
    // The finder ran first and rejected its candidate; the fallback then
    // re-checked every record and configured the applicable two.
    EXPECT_EQ(SingleFinder::state->find_calls.load(), 1);
    EXPECT_EQ(rejected->getState()->applicable_calls.load(), 2);
    EXPECT_EQ(first_applicable->getState()->applicable_calls.load(), 1);
    EXPECT_EQ(second_applicable->getState()->applicable_calls.load(), 1);
    EXPECT_EQ(first_applicable->getState()->create_calls.load(), 1);
    EXPECT_EQ(second_applicable->getState()->create_calls.load(), 1);
    // A finder-less enumeration is not a selection: nothing is cached.
    EXPECT_EQ(cache_observer->getSize(), 0);
}

TEST(OpsEngineTest, ReportsUnsupportedWhenNoFinderProvidesAnApplicableRecord) {
    const PatternBuilder pattern = makeTensorPattern();
    const auto record            = std::make_shared<TestPrimitive>(43, false, 0);
    TestFamily::records          = TestRecords{record};
    armPersona<SingleFinder>(TestRecords{record});
    OpsEngine<TestFamily, SingleFinder> engine;
    const Args args = makeArgs(pattern, engine, {});

    expectStatus(FTRAIN_STATUS_UNSUPPORTED,
                 [&] { static_cast<void>(engine.createPrimitives(args, SelectionContext{getCurrentDeviceId(), 0})); });
}

TEST(OpsEngineTest, RejectsIncompleteOrDifferentPatternArgumentsBeforeSelection) {
    const PatternBuilder pattern = makeTensorPattern();
    const auto record            = std::make_shared<TestPrimitive>(47, true, 0);
    TestFamily::records          = TestRecords{record};
    armPersona<SingleFinder>(TestRecords{record});
    OpsEngine<TestFamily, SingleFinder> engine;

    const Ops ops              = makeOps(pattern, engine);
    const Args incomplete_args = ops.makeArgs();
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] {
        static_cast<void>(engine.createPrimitives(incomplete_args, SelectionContext{getCurrentDeviceId(), 0}));
    });

    PatternBuilder different_pattern;
    const Ops different_ops(different_pattern.buildPattern(), different_pattern.buildPattern());
    const Args different_args = different_ops.makeArgs();
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] {
        static_cast<void>(engine.createPrimitives(different_args, SelectionContext{getCurrentDeviceId(), 0}));
    });

    EXPECT_EQ(SingleFinder::state->enabled_calls.load(), 0);
}

TEST(OpsEngineTest, RejectsInvalidPrimitiveResultsAndFinderCandidates) {
    const PatternBuilder pattern = makeTensorPattern();

    armPersona<SingleFinder>(TestRecords{nullptr});
    TestFamily::records = {};
    OpsEngine<TestFamily, SingleFinder> null_candidate_engine;
    const Args null_candidate_args = makeArgs(pattern, null_candidate_engine, {});
    expectStatus(FTRAIN_STATUS_INTERNAL_ERROR, [&] {
        static_cast<void>(
            null_candidate_engine.createPrimitives(null_candidate_args, SelectionContext{getCurrentDeviceId(), 0}));
    });

    const auto null_record = std::make_shared<TestPrimitive>(53, true, 0, RecordCreation::kNull);
    armPersona<SingleFinder>(TestRecords{null_record});
    TestFamily::records = TestRecords{null_record};
    OpsEngine<TestFamily, SingleFinder> null_record_engine;
    const Args null_record_args = makeArgs(pattern, null_record_engine, {});
    expectStatus(FTRAIN_STATUS_INTERNAL_ERROR, [&] {
        static_cast<void>(
            null_record_engine.createPrimitives(null_record_args, SelectionContext{getCurrentDeviceId(), 0}));
    });

    const auto excessive_workspace = std::make_shared<TestPrimitive>(61, true, 65);
    armPersona<SingleFinder>(TestRecords{excessive_workspace});
    TestFamily::records = TestRecords{excessive_workspace};
    OpsEngine<TestFamily, SingleFinder> excessive_workspace_engine;
    const Args excessive_workspace_args = makeArgs(pattern, excessive_workspace_engine, {});
    expectStatus(FTRAIN_STATUS_INTERNAL_ERROR, [&] {
        static_cast<void>(excessive_workspace_engine.createPrimitives(excessive_workspace_args,
                                                                      SelectionContext{getCurrentDeviceId(), 64}));
    });
}

TEST(OpsEngineTest, PrimitiveAllowListFiltersByOperationalName) {
    const std::unordered_set<std::string> empty;
    const std::unordered_set<std::string> enabled{"Fp32Gemm"};
    const std::unordered_set<std::string> disabled{"Fp32Gemm"};

    EXPECT_TRUE(isPrimitiveAllowed("Fp32Gemm", empty, empty));
    EXPECT_TRUE(isPrimitiveAllowed("Fp32Gemm", enabled, empty));
    EXPECT_FALSE(isPrimitiveAllowed("Fp16Gemm", enabled, empty));
    EXPECT_FALSE(isPrimitiveAllowed("Fp32Gemm", empty, disabled));
    EXPECT_TRUE(isPrimitiveAllowed("Fp16Gemm", empty, disabled));
}

TEST(OpsEngineTest, RejectsNullConstructionDependencies) {
    // A null Finder is unrepresentable: finders are compile-time policies.
    TestFamily::records = TestRecords{nullptr};
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] { OpsEngine<TestFamily, SingleFinder> engine; });

    TestFamily::records = {};
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] { OpsEngine<TestFamily, SingleFinder> engine{nullptr}; });
}

TEST(HandleTest, RegistersAndFindsExactOpsEnginesAndRejectsDuplicates) {
    Handle handle;
    const PatternBuilder tensor_pattern = makeTensorPattern();
    TestFamily::records                 = {};
    const auto tensor_engine            = std::make_shared<OpsEngine<TestFamily, SingleFinder>>();
    const PatternKey tensor_key         = tensor_engine->getPattern().getKey();

    EXPECT_EQ(handle.findOpsEngine(tensor_key), nullptr);
    handle.registerOpsEngine(tensor_engine);
    EXPECT_EQ(handle.getNumOpsEngines(), 1);
    EXPECT_EQ(handle.findOpsEngine(tensor_key), tensor_engine);

    const auto duplicate_engine = std::make_shared<OpsEngine<TestFamily, SingleFinder>>();
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] { handle.registerOpsEngine(duplicate_engine); });
    expectStatus(FTRAIN_STATUS_INVALID_ARGUMENT, [&] { handle.registerOpsEngine(nullptr); });
    EXPECT_EQ(handle.getNumOpsEngines(), 1);
}

TEST(HandleTest, ConcurrentDuplicateRegistrationHasOneWinnerAndLookupRemainsSafe) {
    Handle handle;
    constexpr std::size_t kNumRegistrationThreads = 8;
    constexpr std::size_t kNumLookupThreads       = 4;
    const PatternBuilder pattern                  = makeTensorPattern();
    TestFamily::records                           = {};
    std::vector<std::shared_ptr<OpsEngine<TestFamily, SingleFinder>>> engines;
    engines.reserve(kNumRegistrationThreads);
    for (std::size_t index = 0; index < kNumRegistrationThreads; ++index) {
        engines.push_back(std::make_shared<OpsEngine<TestFamily, SingleFinder>>());
    }
    const PatternKey key = engines.front()->getPattern().getKey();

    std::vector<FTrainStatus> statuses(kNumRegistrationThreads, FTRAIN_STATUS_INTERNAL_ERROR);
    std::vector<std::exception_ptr> errors(kNumRegistrationThreads);
    std::vector<std::thread> threads;
    std::atomic<std::size_t> ready{0};
    std::atomic<std::size_t> registrations_remaining{kNumRegistrationThreads};
    std::atomic<bool> start{false};
    std::atomic<bool> lookup_failed{false};
    threads.reserve(kNumRegistrationThreads + kNumLookupThreads);

    const auto is_candidate = [&](const std::shared_ptr<const OpsEngineBase>& found) {
        return std::any_of(engines.begin(), engines.end(), [&](const auto& engine) { return engine == found; });
    };
    for (std::size_t index = 0; index < kNumRegistrationThreads; ++index) {
        threads.emplace_back([&, index] {
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            try {
                handle.registerOpsEngine(engines[index]);
                statuses[index] = FTRAIN_STATUS_SUCCESS;
            }
            catch (const Exception& exception) {
                statuses[index] = exception.getResult().getStatus();
            }
            catch (...) {
                errors[index] = std::current_exception();
            }
            registrations_remaining.fetch_sub(1, std::memory_order_release);
        });
    }
    for (std::size_t index = 0; index < kNumLookupThreads; ++index) {
        threads.emplace_back([&] {
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            while (registrations_remaining.load(std::memory_order_acquire) != 0) {
                const std::shared_ptr<const OpsEngineBase> found = handle.findOpsEngine(key);
                if ((found != nullptr && !is_candidate(found)) || handle.getNumOpsEngines() > 1) {
                    lookup_failed.store(true, std::memory_order_relaxed);
                }
                std::this_thread::yield();
            }
            for (std::size_t lookup = 0; lookup < 64; ++lookup) {
                const std::shared_ptr<const OpsEngineBase> found = handle.findOpsEngine(key);
                if (found == nullptr || !is_candidate(found) || handle.getNumOpsEngines() != 1) {
                    lookup_failed.store(true, std::memory_order_relaxed);
                }
            }
        });
    }

    const std::size_t num_threads = kNumRegistrationThreads + kNumLookupThreads;
    while (ready.load(std::memory_order_acquire) != num_threads) { std::this_thread::yield(); }
    start.store(true, std::memory_order_release);
    for (std::thread& thread : threads) { thread.join(); }

    for (const std::exception_ptr& error : errors) { EXPECT_FALSE(error); }
    EXPECT_EQ(std::count(statuses.begin(), statuses.end(), FTRAIN_STATUS_SUCCESS), 1);
    EXPECT_EQ(std::count(statuses.begin(), statuses.end(), FTRAIN_STATUS_INVALID_ARGUMENT),
              static_cast<std::ptrdiff_t>(kNumRegistrationThreads - 1));
    EXPECT_FALSE(lookup_failed.load(std::memory_order_relaxed));
    EXPECT_EQ(handle.getNumOpsEngines(), 1);
    const std::shared_ptr<const OpsEngineBase> registered = handle.findOpsEngine(key);
    ASSERT_NE(registered, nullptr);
    EXPECT_TRUE(is_candidate(registered));
}

}  // namespace
}  // namespace ftrain
