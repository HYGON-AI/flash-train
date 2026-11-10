#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <flash_train/flash_train.h>

#include "../support/numerics.hpp"

namespace {

void checkHip(hipError_t status) {
    if (status != hipSuccess) { throw std::runtime_error(hipGetErrorString(status)); }
}

void checkFtrain(FTrainStatus status) {
    if (status != FTRAIN_STATUS_SUCCESS) { throw std::runtime_error(ftrainGetLastMessage()); }
}

uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

float fp32(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct FileBootstrap final {
    std::filesystem::path directory;
    int rank;
    int world_size;
    int sequence = 0;

    void gather(const void* send, void* receive, uint64_t bytes) {
        const auto prefix  = std::to_string(sequence++) + "-";
        const auto target  = directory / (prefix + std::to_string(rank));
        const auto pending = target.string() + ".pending";
        {
            std::ofstream output(pending, std::ios::binary);
            output.write(static_cast<const char*>(send), bytes);
            if (!output.good()) { throw std::runtime_error("Cannot publish bootstrap data"); }
        }
        std::filesystem::rename(pending, target);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        for (int peer = 0; peer < world_size; ++peer) {
            const auto path = directory / (prefix + std::to_string(peer));
            while (!std::filesystem::exists(path)) {
                if (std::chrono::steady_clock::now() > deadline) { throw std::runtime_error("Bootstrap timeout"); }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::ifstream input(path, std::ios::binary);
            input.read(static_cast<char*>(receive) + peer * bytes, bytes);
            if (!input.good() || input.peek() != std::ifstream::traits_type::eof()) {
                throw std::runtime_error("Bootstrap byte count differs between ranks");
            }
        }
    }

    static FTrainStatus callback(void* state, const void* send, void* receive, uint64_t bytes) {
        try {
            static_cast<FileBootstrap*>(state)->gather(send, receive, bytes);
            return FTRAIN_STATUS_SUCCESS;
        }
        catch (const std::exception& error) {
            std::cerr << "Bootstrap failure: " << error.what() << std::endl;
            return FTRAIN_STATUS_INTERNAL_ERROR;
        }
    }

    void barrier() {
        const uint8_t value = 0;
        std::vector<uint8_t> values(world_size);
        gather(&value, values.data(), 1);
    }
};

class DeviceTensor final {
  public:
    DeviceTensor(std::vector<int64_t> dimensions, FTrainNumericType type) : dimensions_(std::move(dimensions)) {
        const auto elements =
            std::accumulate(dimensions_.begin(), dimensions_.end(), int64_t{1}, std::multiplies<int64_t>());
        const auto element_size = type == FTRAIN_NUMERIC_TYPE_INT64   ? sizeof(int64_t)
                                  : type == FTRAIN_NUMERIC_TYPE_INT32 ? sizeof(int32_t)
                                                                      : sizeof(uint16_t);
        bytes_                  = elements * element_size;
        void* memory            = nullptr;
        checkHip(hipMalloc(&memory, bytes_));
        tensor = {
            {memory, dimensions_.data(), nullptr, static_cast<uint8_t>(dimensions_.size()), type,
             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
            memory,
            0
        };
    }

    ~DeviceTensor() { static_cast<void>(hipFree(tensor.storage.memory)); }
    DeviceTensor(const DeviceTensor&)            = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    template<typename Value>
    void upload(const std::vector<Value>& values) {
        if (values.size() * sizeof(Value) != bytes_) { throw std::runtime_error("Invalid upload size"); }
        checkHip(hipMemcpy(tensor.storage.memory, values.data(), bytes_, hipMemcpyHostToDevice));
        ++tensor.version;
    }

    std::vector<uint16_t> download() const {
        std::vector<uint16_t> values(bytes_ / sizeof(uint16_t));
        checkHip(hipMemcpy(values.data(), tensor.storage.memory, bytes_, hipMemcpyDeviceToHost));
        return values;
    }

    FTrainMoeTensor tensor;

  private:
    std::vector<int64_t> dimensions_;
    size_t bytes_;
};

constexpr int active_reductions    = 64;
constexpr float absolute_tolerance = 0.002F;
constexpr float relative_tolerance = 0.02F;

int activePosition(int index, int reduction_size) { return index * (reduction_size - 1) / (active_reductions - 1); }

float inputValue(int row, int active_index) {
    const float sign = (row + active_index) % 3 == 0 ? -1.0F : 1.0F;
    return sign * (row % 5 + 1) * (active_index % 7 + 1) / 64.0F;
}

float weightValue(int expert, int column, int reduction, int factor) {
    return (expert + 1) * (column % 5 + 1) * (reduction % 3 + 1) * factor / 64.0F;
}

float gemmReference(int row, int expert, int column, int factor, int reduction_size) {
    float result = 0;
    for (int index = 0; index < active_reductions; ++index) {
        const auto reduction = activePosition(index, reduction_size);
        result += fp32(bf16(inputValue(row, index))) * fp32(bf16(weightValue(expert, column, reduction, factor)));
    }
    return fp32(bf16(result));
}

struct MoeResources final {
    hipStream_t stream              = nullptr;
    hipStream_t other_stream        = nullptr;
    FTrainCollectiveContext context = nullptr;
    FTrainAGGroupGemm ag            = nullptr;
    FTrainMoEReduceRS rs            = nullptr;
    FTrainPlan plan                 = nullptr;

    ~MoeResources() {
        if (stream != nullptr) { EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess); }
        if (other_stream != nullptr) { EXPECT_EQ(hipStreamSynchronize(other_stream), hipSuccess); }
        if (plan != nullptr) { EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS); }
        if (ag != nullptr) { EXPECT_EQ(ftrainAGGroupGemmDestroy(ag), FTRAIN_STATUS_SUCCESS); }
        if (rs != nullptr) { EXPECT_EQ(ftrainMoEReduceRSDestroy(rs), FTRAIN_STATUS_SUCCESS); }
        if (context != nullptr) { EXPECT_EQ(ftrainCollectiveContextDestroy(context), FTRAIN_STATUS_SUCCESS); }
        if (other_stream != nullptr) { EXPECT_EQ(hipStreamDestroy(other_stream), hipSuccess); }
        if (stream != nullptr) { EXPECT_EQ(hipStreamDestroy(stream), hipSuccess); }
    }
};

struct PlanResources final {
    FTrainPattern pattern = nullptr;
    FTrainOps ops         = nullptr;
    FTrainArgs args       = nullptr;
    FTrainPlan plan       = nullptr;

    ~PlanResources() {
        if (plan != nullptr) { EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS); }
        if (args != nullptr) { EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS); }
        if (ops != nullptr) { EXPECT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS); }
        if (pattern != nullptr) { EXPECT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS); }
    }
};

FTrainPlan buildPlan(FTrainAGGroupGemm ag, FTrainMoEReduceRS rs, const std::vector<FTrainMoeTensor>& tensors) {
    PlanResources resources;
    auto& pattern = resources.pattern;
    auto& ops     = resources.ops;
    auto& args    = resources.args;
    auto& plan    = resources.plan;
    checkFtrain(ftrainPatternCreate(&pattern));
    std::vector<FTrainTensorId> ports(tensors.size());
    for (auto& port : ports) { checkFtrain(ftrainPatternAddTensor(pattern, &port)); }
    FTrainAGGroupGemmOpId ag_op{};
    FTrainMoEReduceRSOpId rs_op{};
    if (ag != nullptr) {
        checkFtrain(ftrainPatternAddAGGroupGemm(pattern, &ag_op, ports[0], ports[1], ports[2], ports[3]));
    } else {
        checkFtrain(ftrainPatternAddMoEReduceRS(pattern, &rs_op, ports[0], ports[1], ports[2], ports[3], ports[4]));
    }
    checkFtrain(ftrainOpsCreate(&ops, pattern));
    checkFtrain(ftrainArgsCreate(&args, ops));
    for (size_t index = 0; index < tensors.size(); ++index) {
        checkFtrain(ftrainArgsSetTensor(args, ports[index], tensors[index].storage));
    }
    if (ag != nullptr) {
        checkFtrain(ftrainArgsSetAGGroupGemm(args, ag_op, ag, tensors[1].version, tensors[2].version));
    } else {
        checkFtrain(ftrainArgsSetMoEReduceRS(args, rs_op, rs, tensors[1].version, tensors[2].version));
    }
    checkFtrain(ftrainPlanCreate(&plan, args));
    uint64_t candidates = 0;
    checkFtrain(ftrainPlanGetNumPrimitives(plan, &candidates));
    if (candidates != 1) { throw std::runtime_error("Expected exactly one explicit route"); }
    return std::exchange(plan, nullptr);
}

void run(FileBootstrap& bootstrap, const std::string& kind, const std::string& route) {
    const bool is_ag    = kind == "ag";
    const bool is_nt    = route.rfind("ag_nt_", 0) == 0 || route == "rs_nt";
    const int experts   = 8;
    const int columns   = is_ag ? 256 : 4096;
    const int reduction = route.find("32k") == std::string::npos ? 1024 : 32768;
    const int topk      = is_ag ? 1 : 2;
    MoeResources resources;
    auto& stream       = resources.stream;
    auto& other_stream = resources.other_stream;
    auto& context      = resources.context;
    auto& ag           = resources.ag;
    auto& rs           = resources.rs;
    checkHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    checkHip(hipStreamCreateWithFlags(&other_stream, hipStreamNonBlocking));
    checkFtrain(ftrainCollectiveContextCreate(&context, bootstrap.rank, bootstrap.world_size, bootstrap.rank,
                                              FileBootstrap::callback, &bootstrap));
    if (is_ag) {
        FTrainAGGroupGemmConfig config{};
        checkFtrain(ftrainAGGroupGemmConfigInit(&config));
        config.m_dim        = 16 * bootstrap.world_size;
        config.capacity_m   = 32 * bootstrap.world_size;
        config.n_dim        = columns;
        config.k_dim        = reduction;
        config.num_experts  = experts;
        config.route        = route.c_str();
        config.m_dim       += bootstrap.rank == 0 ? bootstrap.world_size : 0;
        if (ftrainAGGroupGemmCreate(&ag, context, &config, stream) != FTRAIN_STATUS_INVALID_ARGUMENT || ag != nullptr) {
            throw std::runtime_error("Mismatched AG configuration was not rejected collectively");
        }
        config.m_dim = 16 * bootstrap.world_size;
        checkFtrain(ftrainAGGroupGemmCreate(&ag, context, &config, stream));
    } else {
        FTrainMoEReduceRSConfig config{};
        checkFtrain(ftrainMoEReduceRSConfigInit(&config));
        config.m_dim        = 16 * bootstrap.world_size;
        config.capacity_m   = 32 * bootstrap.world_size;
        config.n_dim        = columns;
        config.k_dim        = reduction;
        config.num_experts  = experts;
        config.topk         = topk;
        config.route        = route.c_str();
        config.m_dim       += bootstrap.rank == 0 ? bootstrap.world_size : 0;
        if (ftrainMoEReduceRSCreate(&rs, context, &config, stream) != FTRAIN_STATUS_INVALID_ARGUMENT || rs != nullptr) {
            throw std::runtime_error("Mismatched RS configuration was not rejected collectively");
        }
        config.m_dim = 16 * bootstrap.world_size;
        checkFtrain(ftrainMoEReduceRSCreate(&rs, context, &config, stream));
    }
    checkFtrain(ftrainCollectiveContextDestroy(context));
    context = nullptr;
    std::vector<int64_t> mismatched_sizes(bootstrap.world_size, 16);
    if (bootstrap.rank == 0) {
        --mismatched_sizes[0];
        ++mismatched_sizes[1];
    }
    if (route != "ag_nt_a2a_chunk") {
        const auto status = is_ag ? ftrainAGGroupGemmSetRankMSizes(ag, 16 * bootstrap.world_size,
                                                                   mismatched_sizes.data(), mismatched_sizes.size())
                                  : ftrainMoEReduceRSSetRankMSizes(rs, 16 * bootstrap.world_size,
                                                                   mismatched_sizes.data(), mismatched_sizes.size());
        if (status != FTRAIN_STATUS_INVALID_ARGUMENT) {
            throw std::runtime_error("Mismatched rank partitions were not rejected collectively");
        }
    }
    DeviceTensor weight(
        is_nt ? std::vector<int64_t>{experts, columns, reduction} : std::vector<int64_t>{experts, reduction, columns},
        FTRAIN_NUMERIC_TYPE_BF16);
    for (int phase = 0; phase < 3; ++phase) {
        std::vector<int64_t> sizes(bootstrap.world_size, 16);
        if (phase == 1 && route == "ag_nt_a2a_chunk") {
            std::fill(sizes.begin(), sizes.end(), 24);
        } else if (phase == 1) {
            for (int peer = 0; peer < bootstrap.world_size; ++peer) { sizes[peer] = 12 + 2 * peer; }
        }
        const int total  = std::accumulate(sizes.begin(), sizes.end(), int64_t{0});
        const int offset = std::accumulate(sizes.begin(), sizes.begin() + bootstrap.rank, int64_t{0});
        if (is_ag) {
            checkFtrain(ftrainAGGroupGemmSetRankMSizes(ag, total, sizes.data(), sizes.size()));
        } else {
            checkFtrain(ftrainMoEReduceRSSetRankMSizes(rs, total, sizes.data(), sizes.size()));
        }
        std::vector<uint16_t> weight_values(size_t(experts) * columns * reduction);
        for (int expert = 0; expert < experts; ++expert) {
            for (int column = 0; column < columns; ++column) {
                for (int inner = 0; inner < reduction; ++inner) {
                    const auto index     = is_nt ? (size_t(expert) * columns + column) * reduction + inner
                                                 : (size_t(expert) * reduction + inner) * columns + column;
                    weight_values[index] = bf16(weightValue(expert, column, inner, bootstrap.rank % 3 + 1 + phase));
                }
            }
        }
        weight.upload(weight_values);
        DeviceTensor input({is_ag ? sizes[bootstrap.rank] : total * topk, reduction}, FTRAIN_NUMERIC_TYPE_BF16);
        const auto routing_type = phase == 1 ? FTRAIN_NUMERIC_TYPE_INT64 : FTRAIN_NUMERIC_TYPE_INT32;
        DeviceTensor routing({total, topk}, routing_type);
        DeviceTensor probabilities({total, topk}, FTRAIN_NUMERIC_TYPE_BF16);
        DeviceTensor output({is_ag ? total : sizes[bootstrap.rank], columns}, FTRAIN_NUMERIC_TYPE_BF16);
        std::vector<int32_t> ids(total * topk);
        std::vector<uint16_t> scales(total * topk);
        if (is_ag) {
            int begin = 0;
            for (const auto size : sizes) {
                for (int row = 0; row < size; ++row) { ids[begin + row] = (row + phase) * experts / (size + phase); }
                begin += size;
            }
        } else {
            for (int row = 0; row < total; ++row) {
                for (int choice = 0; choice < topk; ++choice) {
                    ids[row * topk + choice]    = (row + choice + phase) % experts;
                    scales[row * topk + choice] = bf16((choice + 1) / 4.0F);
                }
            }
        }
        const int input_rows = is_ag ? sizes[bootstrap.rank] : total * topk;
        std::vector<uint16_t> input_values(size_t(input_rows) * reduction);
        for (int row = 0; row < input_rows; ++row) {
            const int original_row = is_ag ? row + offset : row / topk;
            for (int index = 0; index < active_reductions; ++index) {
                input_values[size_t(row) * reduction + activePosition(index, reduction)] =
                    bf16(inputValue(original_row + phase, index));
            }
        }
        input.upload(input_values);
        if (routing_type == FTRAIN_NUMERIC_TYPE_INT64) {
            routing.upload(std::vector<int64_t>(ids.begin(), ids.end()));
        } else {
            routing.upload(ids);
        }
        routing.tensor.version = phase + 1;
        probabilities.upload(scales);
        std::vector<FTrainMoeTensor> tensors{input.tensor, weight.tensor, routing.tensor};
        if (!is_ag) { tensors.push_back(probabilities.tensor); }
        tensors.push_back(output.tensor);
        resources.plan      = buildPlan(ag, rs, tensors);
        const auto plan     = resources.plan;
        float phase_maximum = 0;
        for (int repeat = 0; repeat < 3; ++repeat) {
            for (const bool staged : {false, true}) {
                const std::string path     = staged ? "plan" : "convenience";
                const auto selected_stream = staged ? other_stream : stream;
                output.upload(std::vector<uint16_t>(size_t(is_ag ? total : sizes[bootstrap.rank]) * columns, 0x7fc0));
                if (staged) {
                    checkFtrain(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, selected_stream));
                } else if (is_ag) {
                    checkFtrain(ftrainAGGroupGemmExecute(ag, input.tensor, weight.tensor, routing.tensor, output.tensor,
                                                         selected_stream));
                } else {
                    checkFtrain(ftrainMoEReduceRSExecute(rs, input.tensor, weight.tensor, routing.tensor,
                                                         probabilities.tensor, output.tensor, selected_stream));
                }
                checkHip(hipStreamSynchronize(selected_stream));
                const auto actual = output.download();
                std::vector<float> observed_values(actual.size());
                std::vector<float> expected_values(actual.size());
                float maximum = 0;
                for (int local_row = 0; local_row < (is_ag ? total : sizes[bootstrap.rank]); ++local_row) {
                    const int row = is_ag ? local_row : local_row + offset;
                    for (int column = 0; column < columns; ++column) {
                        float expected = 0;
                        if (is_ag) {
                            expected =
                                gemmReference(row + phase, ids[row], column, bootstrap.rank % 3 + 1 + phase, reduction);
                        } else {
                            for (int peer = 0; peer < bootstrap.world_size; ++peer) {
                                for (int choice = 0; choice < topk; ++choice) {
                                    expected += gemmReference(row + phase, ids[row * topk + choice], column,
                                                              peer % 3 + 1 + phase, reduction) *
                                                (choice + 1) / 4.0F;
                                }
                            }
                            expected = fp32(bf16(expected));
                        }
                        const auto observed = fp32(actual[size_t(local_row) * columns + column]);
                        const auto error    = std::abs(observed - expected);
                        observed_values[size_t(local_row) * columns + column] = observed;
                        expected_values[size_t(local_row) * columns + column] = expected;
                        maximum                                               = std::max(maximum, error);
                    }
                }
                SCOPED_TRACE("route=" + route + " rank=" + std::to_string(bootstrap.rank) +
                             " phase=" + std::to_string(phase) + " repeat=" + std::to_string(repeat) + " path=" + path);
                EXPECT_TRUE(std::all_of(observed_values.begin(), observed_values.end(), [](float value) {
                    return std::isfinite(value);
                })) << "Output contains non-finite values";
                ftrain_test::expectNear(observed_values, expected_values, relative_tolerance, absolute_tolerance,
                                        path.c_str());
                phase_maximum = std::max(phase_maximum, maximum);
                std::cout << "CHECK " << route << " phase=" << phase << " repeat=" << repeat << " path=" << path
                          << " max_abs=" << maximum << std::endl;
            }
        }
        if (is_ag) {
            checkFtrain(ftrainAGGroupGemmSynchronize(ag));
        } else {
            checkFtrain(ftrainMoEReduceRSSynchronize(rs));
        }
        checkFtrain(ftrainPlanDestroy(plan));
        resources.plan = nullptr;
        std::cout << "CHECKED " << route << " phase=" << phase << " max_abs=" << phase_maximum << std::endl;
        bootstrap.barrier();
    }
    bootstrap.barrier();
}

std::string unavailableDevices(int world_size) {
    int available     = 0;
    const auto status = hipGetDeviceCount(&available);
    if (status == hipErrorNoDevice || (status == hipSuccess && available < world_size)) {
        return "requires " + std::to_string(world_size) + " visible gfx936 devices, found " + std::to_string(available);
    }
    checkHip(status);
    for (int device = 0; device < world_size; ++device) {
        hipDeviceProp_t properties{};
        checkHip(hipGetDeviceProperties(&properties, device));
        if (std::string(properties.gcnArchName).substr(0, 6) != "gfx936") {
            return "device " + std::to_string(device) + " is not gfx936";
        }
        checkHip(hipSetDevice(device));
        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        checkHip(hipMemGetInfo(&free_bytes, &total_bytes));
        if (free_bytes < (size_t{2} << 30)) { return "device " + std::to_string(device) + " has less than 2 GiB free"; }
    }
    return {};
}

int environmentInt(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) { throw std::runtime_error(std::string("Missing environment variable: ") + name); }
    size_t consumed  = 0;
    const int result = std::stoi(value, &consumed);
    if (consumed != std::strlen(value)) { throw std::runtime_error(std::string("Invalid integer: ") + name); }
    return result;
}

TEST(MoeNumericsEnvironmentTest, DevicesAvailable) {
    if (std::getenv("FTRAIN_TEST_MOE_WORLD_SIZE") == nullptr) {
        GTEST_SKIP() << "Run through tests/support/run_multirank.py --gtest";
    }
    const int world_size = environmentInt("FTRAIN_TEST_MOE_WORLD_SIZE");
    ASSERT_GE(world_size, 2);
    ASSERT_LE(world_size, 8);
    const auto reason = unavailableDevices(world_size);
    if (!reason.empty()) { GTEST_SKIP() << reason; }
}

class MoeNumericsTest : public testing::TestWithParam<const char*> {
  protected:
    void SetUp() override {
        const char* directory = std::getenv("FTRAIN_TEST_MOE_RENDEZVOUS");
        if (directory == nullptr) { GTEST_SKIP() << "Requires the multi-rank GoogleTest launcher"; }
        const int rank       = environmentInt("FTRAIN_TEST_MOE_RANK");
        const int world_size = environmentInt("FTRAIN_TEST_MOE_WORLD_SIZE");
        ASSERT_GE(world_size, 2);
        ASSERT_LE(world_size, 8);
        ASSERT_GE(rank, 0);
        ASSERT_LT(rank, world_size);
        ASSERT_TRUE(std::filesystem::is_directory(directory));
        ASSERT_EQ(hipSetDevice(rank), hipSuccess);
        bootstrap = std::make_unique<FileBootstrap>(FileBootstrap{directory, rank, world_size});
        RecordProperty("rank", rank);
        RecordProperty("world_size", world_size);
        RecordProperty("route", GetParam());
    }

    std::unique_ptr<FileBootstrap> bootstrap;
};

TEST_P(MoeNumericsTest, MatchesHostReference) {
    const std::string route = GetParam();
    ASSERT_NO_THROW(run(*bootstrap, route.rfind("ag_", 0) == 0 ? "ag" : "rs", route));
}

INSTANTIATE_TEST_SUITE_P(Routes, MoeNumericsTest,
                         testing::Values("ag_nn_a2a", "ag_nn_ring", "ag_nn_ring_chunk", "ag_nt_a2a", "ag_nt_a2a_chunk",
                                         "ag_nt_ring_chunk", "ag_nt_a2a_chunk_32k", "ag_nt_ring_chunk_32k", "rs_nn",
                                         "rs_nt"),
                         [](const testing::TestParamInfo<const char*>& parameter) {
                             return std::string(parameter.param);
                         });

}  // namespace
