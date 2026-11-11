#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hip/hip_runtime.h>
#include <flash_train/flash_train.h>

namespace {

constexpr int kExperts = 64;
constexpr int kTopk = 6;
constexpr int kSamples = 128;

void checkHip(hipError_t status) {
    if (status != hipSuccess) { throw std::runtime_error(hipGetErrorString(status)); }
}

void checkFtrain(FTrainStatus status) {
    if (status != FTRAIN_STATUS_SUCCESS) { throw std::runtime_error(ftrainGetLastMessage()); }
}

void prepareSession(FTrainCollectiveContext context, bool is_ag, const FTrainAGGroupGemmConfig& ag_config,
                    FTrainStream stream) {
    FTrainPattern pattern_handle = nullptr;
    checkFtrain(ftrainPatternCreate(&pattern_handle));
    std::unique_ptr<FTrainPatternStruct, decltype(&ftrainPatternDestroy)> pattern(pattern_handle, ftrainPatternDestroy);
    std::vector<FTrainTensorId> ports(is_ag ? 4 : 5);
    for (auto& port : ports) { checkFtrain(ftrainPatternAddTensor(pattern.get(), &port)); }
    FTrainAGGroupGemmOpId ag_op{};
    {
        checkFtrain(ftrainPatternAddAGGroupGemm(pattern.get(), &ag_op, ports[0], ports[1], ports[2], ports[3]));
    }
    FTrainOps ops_handle = nullptr;
    checkFtrain(ftrainOpsCreate(&ops_handle, pattern.get()));
    std::unique_ptr<FTrainOpsStruct, decltype(&ftrainOpsDestroy)> ops(ops_handle, ftrainOpsDestroy);
    FTrainArgs args_handle = nullptr;
    checkFtrain(ftrainArgsCreate(&args_handle, ops.get()));
    std::unique_ptr<FTrainArgsStruct, decltype(&ftrainArgsDestroy)> args(args_handle, ftrainArgsDestroy);
    {
        checkFtrain(ftrainArgsSetAGGroupGemm(args.get(), ag_op, context, &ag_config, 0, 0, stream));
    }
}

__host__ __device__ uint32_t mixBits(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__host__ __device__ float inputValue(int row, int inner, int rank) {
    return (int(mixBits(uint32_t(row) * 65537U + uint32_t(inner) * 257U + uint32_t(rank) * 1009U) % 63) - 31) / 2048.0F;
}

__host__ __device__ float weightValue(int expert, int column, int inner, int rank) {
    return (int(mixBits(uint32_t(expert) * 65537U + uint32_t(column) * 1009U +
                        uint32_t(inner) * 257U + uint32_t(rank) * 131U + 17U) % 63) - 31) / 2048.0F;
}

__host__ __device__ uint16_t bf16(float value) {
    union { float floating; uint32_t integer; } bits;
    bits.floating = value;
    bits.integer += 0x7fff + ((bits.integer >> 16) & 1);
    return uint16_t(bits.integer >> 16);
}

__host__ __device__ float fp32(uint16_t value) {
    union { float floating; uint32_t integer; } bits;
    bits.integer = uint32_t(value) << 16;
    return bits.floating;
}

__global__ void fillInput(uint16_t* output, uint64_t count, int reduction, const int32_t* rows, int rank) {
    for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += uint64_t(gridDim.x) * blockDim.x) {
        output[index] = bf16(inputValue(rows[index / reduction], index % reduction, rank));
    }
}

__global__ void fillWeight(uint16_t* output, uint64_t count, int columns, int reduction, int rank, bool nt) {
    const uint64_t matrix_size = uint64_t(columns) * reduction;
    for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t within = index % matrix_size;
        const int column = nt ? within / reduction : within % columns;
        const int inner = nt ? within % reduction : within / columns;
        output[index] = bf16(weightValue(index / matrix_size, column, inner, rank));
    }
}

__global__ void inspectOutput(const uint16_t* output, uint64_t count, int rows, int columns,
                              int32_t* nonfinite, uint16_t* samples) {
    for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += uint64_t(gridDim.x) * blockDim.x) {
        if ((output[index] & 0x7f80) == 0x7f80) { atomicAdd(nonfinite, 1); }
    }
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < kSamples) {
        const uint64_t row = uint64_t(sample) * (rows - 1) / (kSamples - 1);
        const int column = (sample * 271 + 17) % columns;
        samples[sample] = output[row * columns + column];
    }
}

__global__ void sampleStorage(const uint16_t* data, uint64_t count, uint16_t* samples) {
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < 1024) { samples[sample] = data[uint64_t(sample) * (count - 1) / 1023]; }
}

struct Bootstrap final {
    std::filesystem::path directory;
    int rank;
    int world;
    int sequence = 0;

    void gather(const void* send, void* receive, uint64_t bytes) {
        const auto prefix = std::to_string(sequence++) + "-";
        const auto target = directory / (prefix + std::to_string(rank));
        const auto pending = target.string() + ".pending";
        {
            std::ofstream output(pending, std::ios::binary);
            output.write(static_cast<const char*>(send), bytes);
            if (!output.good()) { throw std::runtime_error("Bootstrap write failed"); }
        }
        std::filesystem::rename(pending, target);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        for (int peer = 0; peer < world; ++peer) {
            const auto path = directory / (prefix + std::to_string(peer));
            while (!std::filesystem::exists(path)) {
                if (std::chrono::steady_clock::now() > deadline) { throw std::runtime_error("Bootstrap timeout"); }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::ifstream input(path, std::ios::binary);
            input.read(static_cast<char*>(receive) + peer * bytes, bytes);
            if (!input.good() || input.peek() != std::ifstream::traits_type::eof()) {
                throw std::runtime_error("Bootstrap byte count mismatch");
            }
        }
    }

    static FTrainStatus callback(void* state, const void* send, void* receive, uint64_t bytes) {
        try {
            static_cast<Bootstrap*>(state)->gather(send, receive, bytes);
            return FTRAIN_STATUS_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            return FTRAIN_STATUS_INTERNAL_ERROR;
        }
    }

    void barrier() {
        const uint8_t value = 0;
        std::vector<uint8_t> values(world);
        gather(&value, values.data(), 1);
    }
};

class DeviceTensor final {
  public:
    DeviceTensor(std::vector<int64_t> dimensions, FTrainNumericType type) : dimensions_(std::move(dimensions)) {
        elements_ = std::accumulate(dimensions_.begin(), dimensions_.end(), int64_t{1}, std::multiplies<int64_t>());
        bytes_ = elements_ * (type == FTRAIN_NUMERIC_TYPE_INT32 ? sizeof(int32_t) : sizeof(uint16_t));
        void* memory = nullptr;
        checkHip(hipMalloc(&memory, bytes_));
        tensor = {{memory, dimensions_.data(), nullptr, uint8_t(dimensions_.size()), type,
                    FTRAIN_INDEX_TYPE_CONTINUOUS, false}, memory, 1};
    }

    ~DeviceTensor() { static_cast<void>(hipFree(tensor.storage.memory)); }
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    template<typename Value> Value* data() const { return static_cast<Value*>(tensor.storage.memory); }
    template<typename Value> void upload(const std::vector<Value>& values) {
        if (values.size() * sizeof(Value) != bytes_) { throw std::runtime_error("Upload size mismatch"); }
        checkHip(hipMemcpy(tensor.storage.memory, values.data(), bytes_, hipMemcpyHostToDevice));
    }
    uint64_t elements() const { return elements_; }
    uint64_t bytes() const { return bytes_; }
    FTrainMoeTensor tensor;

  private:
    std::vector<int64_t> dimensions_;
    uint64_t elements_;
    uint64_t bytes_;
};

uint64_t fingerprint(const DeviceTensor& tensor, hipStream_t stream) {
    DeviceTensor sampled({1024}, FTRAIN_NUMERIC_TYPE_BF16);
    sampleStorage<<<4, 256, 0, stream>>>(tensor.data<uint16_t>(), tensor.bytes() / 2, sampled.data<uint16_t>());
    checkHip(hipGetLastError());
    checkHip(hipStreamSynchronize(stream));
    std::vector<uint16_t> values(1024);
    checkHip(hipMemcpy(values.data(), sampled.data<uint16_t>(), sampled.bytes(), hipMemcpyDeviceToHost));
    uint64_t hash = 14695981039346656037ULL;
    for (const auto value : values) { hash = (hash ^ value) * 1099511628211ULL; }
    return hash;
}

void writeValues(std::ostream& output, const std::vector<float>& values) {
    output << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) { output << ','; }
        output << values[index];
    }
    output << ']';
}

void run(const std::string& route, Bootstrap& bootstrap, int logical_m, int global_n, int global_k,
         int warmup, int iterations, bool ragged, bool balanced_topk) {
    if (logical_m <= 0 || global_n <= 0 || global_k <= 0) { throw std::runtime_error("Dimensions must be positive"); }
    const bool is_ag = route.rfind("ag_", 0) == 0;
    if (!is_ag) { throw std::runtime_error("Only AG routes are available"); }
    const bool is_nt = route.rfind("ag_nt_", 0) == 0;
    const int columns = is_ag ? global_n / bootstrap.world : global_n;
    const int reduction = is_ag ? global_k : global_k / bootstrap.world;
    const int operation_m = is_ag ? logical_m * kTopk : logical_m;
    std::vector<int64_t> logical_sizes(bootstrap.world, logical_m / bootstrap.world);
    for (int peer = 0; peer < bootstrap.world; ++peer) {
        if (ragged) { logical_sizes[peer] += (2 * peer - bootstrap.world + 1) * 16; }
        if (logical_sizes[peer] <= 0) { throw std::runtime_error("Partition is too small"); }
    }
    const auto minimum = *std::min_element(logical_sizes.begin(), logical_sizes.end());
    if (logical_m % bootstrap.world != 0 || global_n % bootstrap.world != 0 ||
        global_k % bootstrap.world != 0 || minimum <= 0) { throw std::runtime_error("Invalid dimensions"); }
    std::vector<int64_t> offsets(bootstrap.world + 1, 0);
    std::partial_sum(logical_sizes.begin(), logical_sizes.end(), offsets.begin() + 1);
    auto rank_sizes = logical_sizes;
    { for (auto& size : rank_sizes) { size *= kTopk; } }
    const int input_rows = is_ag ? rank_sizes[bootstrap.rank] : logical_m * kTopk;
    const int output_rows = is_ag ? operation_m : logical_sizes[bootstrap.rank];
    std::vector<int32_t> routing_values(size_t(logical_m) * kTopk);
    std::vector<int32_t> source_rows(is_ag ? operation_m : input_rows);
    {
        size_t destination = 0;
        for (int peer = 0; peer < bootstrap.world; ++peer) {
            for (int expert = 0; expert < kExperts; ++expert) {
                for (int local = 0; local < logical_sizes[peer] * kTopk; ++local) {
                    const int expanded = offsets[peer] * kTopk + local;
                    const int assigned_expert = balanced_topk ? expanded % kExperts :
                        mixBits(uint32_t(expanded) + 3000U) % kExperts;
                    if (assigned_expert == expert) {
                        routing_values[destination] = expert;
                        source_rows[destination++] = expanded / kTopk;
                    }
                }
            }
        }
    }
    hipStream_t stream = nullptr;
    checkHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    DeviceTensor input({input_rows, reduction}, FTRAIN_NUMERIC_TYPE_BF16);
    DeviceTensor weight(is_nt ? std::vector<int64_t>{kExperts, columns, reduction} :
                               std::vector<int64_t>{kExperts, reduction, columns}, FTRAIN_NUMERIC_TYPE_BF16);
    DeviceTensor routing({is_ag ? operation_m : logical_m, is_ag ? 1 : kTopk}, FTRAIN_NUMERIC_TYPE_INT32);
    DeviceTensor probabilities({logical_m, kTopk}, FTRAIN_NUMERIC_TYPE_BF16);
    DeviceTensor output({output_rows, columns}, FTRAIN_NUMERIC_TYPE_BF16);
    DeviceTensor device_rows({input_rows}, FTRAIN_NUMERIC_TYPE_INT32);
    {
        const size_t begin = offsets[bootstrap.rank] * kTopk;
        device_rows.upload(std::vector<int32_t>(source_rows.begin() + begin, source_rows.begin() + begin + input_rows));
    }
    routing.upload(routing_values);
    std::vector<uint16_t> probability_values(size_t(logical_m) * kTopk);
    for (size_t index = 0; index < probability_values.size(); ++index) {
        probability_values[index] = bf16(float(index % kTopk + 1) / 21.0F);
    }
    probabilities.upload(probability_values);
    fillInput<<<4096, 256, 0, stream>>>(input.data<uint16_t>(), input.elements(), reduction,
                                       device_rows.data<int32_t>(), bootstrap.rank);
    fillWeight<<<8192, 256, 0, stream>>>(weight.data<uint16_t>(), weight.elements(), columns,
                                        reduction, bootstrap.rank, is_nt);
    checkHip(hipGetLastError());
    checkHip(hipMemsetAsync(output.tensor.storage.memory, 0xff, output.bytes(), stream));
    checkHip(hipStreamSynchronize(stream));
    const std::vector<uint64_t> fingerprints{fingerprint(input, stream), fingerprint(weight, stream),
                                            fingerprint(routing, stream), fingerprint(probabilities, stream)};
    FTrainCollectiveContext context = nullptr;
    checkFtrain(ftrainCollectiveContextCreate(&context, bootstrap.rank, bootstrap.world, bootstrap.rank,
                                              Bootstrap::callback, &bootstrap));
    FTrainAGGroupGemmConfig ag_config = FTRAIN_AG_GROUP_GEMM_CONFIG_INIT;
    {
        auto& config = ag_config;
        config.m_dim = operation_m;
        config.n_dim = columns;
        config.k_dim = reduction;
        config.num_experts = kExperts;
        config.topk = 1;
        config.rank_m_sizes = rank_sizes.data();
        config.num_rank_m_sizes = rank_sizes.size();
        config.route = route.c_str();
    }
    prepareSession(context, is_ag, ag_config, stream);
    auto execute = [&] {
        {
            checkFtrain(ftrainAGGroupGemm(context, &ag_config, input.tensor, weight.tensor, routing.tensor, output.tensor, stream));
        }
    };
    hipEvent_t started = nullptr;
    hipEvent_t ended = nullptr;
    checkHip(hipEventCreate(&started));
    checkHip(hipEventCreate(&ended));
    std::vector<float> device_samples;
    std::vector<float> host_samples;
    bootstrap.barrier();
    for (int iteration = 0; iteration < warmup + iterations; ++iteration) {
        checkHip(hipEventRecord(started, stream));
        const auto before = std::chrono::steady_clock::now();
        execute();
        const auto after = std::chrono::steady_clock::now();
        checkHip(hipEventRecord(ended, stream));
        checkHip(hipEventSynchronize(ended));
        float milliseconds = 0;
        checkHip(hipEventElapsedTime(&milliseconds, started, ended));
        if (iteration >= warmup) {
            device_samples.push_back(milliseconds);
            host_samples.push_back(std::chrono::duration<float, std::milli>(after - before).count());
        }
    }
    bootstrap.barrier();
    DeviceTensor invalid({1}, FTRAIN_NUMERIC_TYPE_INT32);
    DeviceTensor sampled({kSamples}, FTRAIN_NUMERIC_TYPE_BF16);
    checkHip(hipMemsetAsync(invalid.tensor.storage.memory, 0, invalid.bytes(), stream));
    inspectOutput<<<4096, 256, 0, stream>>>(output.data<uint16_t>(), output.elements(), output_rows, columns,
                                           invalid.data<int32_t>(), sampled.data<uint16_t>());
    checkHip(hipGetLastError());
    checkHip(hipStreamSynchronize(stream));
    int nonfinite = 0;
    std::vector<uint16_t> actual(kSamples);
    checkHip(hipMemcpy(&nonfinite, invalid.tensor.storage.memory, sizeof(nonfinite), hipMemcpyDeviceToHost));
    checkHip(hipMemcpy(actual.data(), sampled.tensor.storage.memory, sampled.bytes(), hipMemcpyDeviceToHost));
    std::vector<float> expected(kSamples, 0);
    {
        for (int sample = 0; sample < kSamples; ++sample) {
            const int row = int64_t(sample) * (operation_m - 1) / (kSamples - 1);
            const int owner = std::upper_bound(offsets.begin(), offsets.end(), row / kTopk) - offsets.begin() - 1;
            const int column = (sample * 271 + 17) % columns;
            for (int inner = 0; inner < reduction; ++inner) {
                expected[sample] += inputValue(source_rows[row], inner, owner) *
                                    weightValue(routing_values[row], column, inner, bootstrap.rank);
            }
        }
    }
    float maximum = 0;
    bool passed = nonfinite == 0;
    std::vector<float> actual_values;
    for (int sample = 0; sample < kSamples; ++sample) {
        const float value = fp32(actual[sample]);
        const float error = std::abs(value - expected[sample]);
        maximum = std::max(maximum, error);
        passed = passed && std::isfinite(value) && error <= 0.0005F + 0.025F * std::abs(expected[sample]);
        actual_values.push_back(value);
    }
    Dl_info module{};
    if (dladdr(reinterpret_cast<const void*>(&ftrainMoeIsAvailable), &module) == 0) {
        throw std::runtime_error("Cannot determine loaded library");
    }
    std::vector<std::string> libraries;
    std::ifstream maps("/proc/self/maps");
    std::string mapping;
    while (std::getline(maps, mapping)) {
        const auto path_begin = mapping.find('/');
        if (path_begin != std::string::npos && mapping.find(".so", path_begin) != std::string::npos) {
            libraries.push_back(std::filesystem::canonical(mapping.substr(path_begin)).string());
        }
    }
    std::sort(libraries.begin(), libraries.end());
    libraries.erase(std::unique(libraries.begin(), libraries.end()), libraries.end());
    const auto result_path = bootstrap.directory.parent_path() / ("rank-" + std::to_string(bootstrap.rank) + ".json");
    std::ofstream record(result_path);
    record << std::setprecision(9) << "{\"route\":\"" << route << "\",\"rank\":" << bootstrap.rank
           << ",\"world\":" << bootstrap.world << ",\"logical_m\":" << logical_m
           << ",\"global_n\":" << global_n << ",\"global_k\":" << global_k
           << ",\"operator_m\":" << operation_m << ",\"operator_n\":" << columns
           << ",\"operator_k\":" << reduction << ",\"experts\":" << kExperts
           << ",\"logical_topk\":" << kTopk << ",\"operator_topk\":" << (is_ag ? 1 : kTopk)
           << ",\"ragged\":" << (ragged ? "true" : "false")
           << ",\"balanced_topk\":" << (is_ag && balanced_topk ? "true" : "false") << ",\"unique_topk\":false"
           << ",\"library\":\"" << std::filesystem::canonical(module.dli_fname).string()
           << "\",\"passed\":" << (passed ? "true" : "false") << ",\"nonfinite\":" << nonfinite
           << ",\"max_abs\":" << maximum << ",\"fingerprints\":[";
    for (size_t index = 0; index < fingerprints.size(); ++index) {
        if (index != 0) { record << ','; }
        record << '"' << fingerprints[index] << '"';
    }
    std::vector<int64_t> expert_counts(kExperts, 0);
    for (const auto expert : routing_values) { ++expert_counts[expert]; }
    record << "],\"expert_counts\":[";
    for (size_t expert = 0; expert < expert_counts.size(); ++expert) {
        if (expert != 0) { record << ','; }
        record << expert_counts[expert];
    }
    record << "],\"gpu_ms\":";
    writeValues(record, device_samples);
    record << ",\"host_enqueue_ms\":";
    writeValues(record, host_samples);
    record << ",\"output_samples\":";
    writeValues(record, actual_values);
    record << ",\"mapped_libraries\":[";
    for (size_t index = 0; index < libraries.size(); ++index) {
        if (index != 0) { record << ','; }
        record << '"' << libraries[index] << '"';
    }
    record << ']';
    record << "}\n";
    record.close();
    if (!record.good()) { throw std::runtime_error("Cannot write benchmark result"); }
    bootstrap.barrier();
    checkFtrain(ftrainCollectiveContextDestroy(context));
    checkHip(hipEventDestroy(started));
    checkHip(hipEventDestroy(ended));
    checkHip(hipStreamDestroy(stream));
    if (!passed) { throw std::runtime_error("Numerical validation failed"); }
    std::cout << "PASS " << route << " rank=" << bootstrap.rank << " max_abs=" << maximum << std::endl;
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 11 && argc != 12) {
            throw std::runtime_error("usage: ftrain_moe_perf route rendezvous rank world M N K warmup iters equal|ragged [balanced_topk=true|false]");
        }
        const std::string balanced_topk = argc == 12 ? argv[11] : "false";
        if (balanced_topk != "true" && balanced_topk != "false") {
            throw std::runtime_error("balanced_topk must be true or false");
        }
        Bootstrap bootstrap{argv[2], std::stoi(argv[3]), std::stoi(argv[4])};
        if (bootstrap.world != 8 || bootstrap.rank < 0 || bootstrap.rank >= bootstrap.world ||
            (std::string(argv[10]) != "equal" && std::string(argv[10]) != "ragged")) {
            throw std::runtime_error("Expected eight ranks and equal|ragged partition");
        }
        checkHip(hipSetDevice(bootstrap.rank));
        const int warmup = std::stoi(argv[8]);
        const int iterations = std::stoi(argv[9]);
        if (warmup < 0 || iterations <= 0) { throw std::runtime_error("Invalid iteration count"); }
        run(argv[1], bootstrap, std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]),
            warmup, iterations, std::string(argv[10]) == "ragged", balanced_topk == "true");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
