#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>

#include <hip/hip_runtime.h>
#include <flash_train/flash_train.h>

#include "op.hpp"
#include "context.hpp"

namespace ftrain_bench {
namespace {

const char* requiredEnvironment(const char* name) {
    const char* value = std::getenv(name);
    require(value != nullptr && *value != '\0',
            (std::string("missing or empty MoE benchmark environment: ") + name +
             "; launch through ftrain_bench cbench").c_str());
    return value;
}

__host__ __device__ uint32_t mixBits(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__host__ __device__ uint16_t bf16(float value) {
    union {
        float floating;
        uint32_t integer;
    } bits;
    bits.floating  = value;
    bits.integer  += 0x7fff + ((bits.integer >> 16) & 1);
    return uint16_t(bits.integer >> 16);
}

float fp32(uint16_t value) {
    union {
        float floating;
        uint32_t integer;
    } bits;
    bits.integer = uint32_t(value) << 16;
    return bits.floating;
}

__host__ __device__ float inputValue(int row, int inner, int rank, uint32_t salt) {
    return (int(mixBits(uint32_t(row) * 65537U + uint32_t(inner) * 257U + uint32_t(rank) * 1009U + salt) % 63) - 31) /
           2048.0F;
}

__host__ __device__ float weightValue(int expert, int column, int inner, int rank, uint32_t salt) {
    return (int(mixBits(uint32_t(expert) * 65537U + uint32_t(column) * 1009U + uint32_t(inner) * 257U +
                        uint32_t(rank) * 131U + 17U + salt) %
                63) -
            31) /
           2048.0F;
}

__global__ void fillInput(uint16_t* output, uint64_t count, int reduction, const int32_t* rows, int rank,
                          uint32_t salt) {
    for (uint64_t index  = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index          += uint64_t(gridDim.x) * blockDim.x) {
        output[index] = bf16(inputValue(rows[index / reduction], index % reduction, rank, salt));
    }
    return;
}

__global__ void fillWeight(uint16_t* output, uint64_t count, int columns, int reduction, int rank, bool nt,
                           uint32_t salt) {
    const uint64_t matrix_size = uint64_t(columns) * reduction;
    for (uint64_t index  = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index          += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t within = index % matrix_size;
        const int column      = nt ? within / reduction : within % columns;
        const int inner       = nt ? within % reduction : within / columns;
        output[index]         = bf16(weightValue(index / matrix_size, column, inner, rank, salt));
    }
    return;
}

__global__ void inspect(const uint16_t* output, uint64_t count, int* invalid) {
    for (uint64_t index  = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index          += uint64_t(gridDim.x) * blockDim.x) {
        if ((output[index] & 0x7f80) == 0x7f80) { atomicExch(invalid, 1); }
    }
    return;
}

struct DeviceTensor final {
    std::vector<int64_t> dims;
    uint64_t elements;
    uint64_t bytes;
    FTrainMoeTensor tensor{};

    DeviceTensor(std::vector<int64_t> dimensions, FTrainNumericType type)
        : dims(std::move(dimensions)),
          elements(std::accumulate(dims.begin(), dims.end(), uint64_t{1}, std::multiplies<uint64_t>{})),
          bytes(elements * (type == FTRAIN_NUMERIC_TYPE_INT32 ? 4 : 2)) {
        void* memory = nullptr;
        checkHip(hipMalloc(&memory, bytes), "MoE benchmark tensor allocation");
        tensor = {
            {memory, dims.data(), nullptr, static_cast<uint8_t>(dims.size()), type, FTRAIN_INDEX_TYPE_CONTINUOUS,
             false},
            memory,
            0
        };
    }

    ~DeviceTensor() { checkHip(hipFree(tensor.storage.memory), "free benchmark tensor"); }
    DeviceTensor(const DeviceTensor&)            = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    template<typename Value>
    Value* data() const {
        return static_cast<Value*>(tensor.storage.memory);
    }

    std::string fingerprint() const {
        uint64_t hash = 14695981039346656037ULL;
        for (uint64_t sample = 0; sample < 128; ++sample) {
            uint8_t value       = 0;
            const auto* address = data<uint8_t>() + sample * (bytes - 1) / 127;
            checkHip(hipMemcpy(&value, address, 1, hipMemcpyDeviceToHost), "fingerprint read");
            hash = (hash ^ value) * 1099511628211ULL;
        }
        std::ostringstream output;
        output << '"' << std::hex << std::setw(16) << std::setfill('0') << hash << '"';
        return output.str();
    }
};

struct MoeSession final : OpSession {
    MoeContext bootstrap;
    bool is_ag;
    bool is_nt;
    std::string route;
    int logical_m;
    int reduction;
    int columns;
    int topk      = environmentInt("FTRAIN_BENCH_TOPK", 6);
    int experts   = environmentInt("FTRAIN_BENCH_EXPERTS", 64);
    uint32_t salt = static_cast<uint32_t>(std::stoull(requiredEnvironment("FTRAIN_BENCH_SEED"))) - 2026U;
    std::vector<int64_t> sizes;
    std::vector<int64_t> offsets;
    std::vector<int32_t> routing_values;
    std::vector<int32_t> source_rows;
    std::vector<std::unique_ptr<DeviceTensor>> tensors;
    std::vector<FTrainTensorId> ports;
    FTrainCollectiveContext context = nullptr;
    FTrainAGGroupGemm ag            = nullptr;
    FTrainMoEReduceRS rs            = nullptr;
    FTrainAGGroupGemmOpId ag_op{};
    FTrainMoEReduceRSOpId rs_op{};
    std::string fingerprints;

    MoeSession(bool ag_kind, const std::vector<int64_t>& shape)
        : is_ag(ag_kind), is_nt(false), route(requiredEnvironment("FTRAIN_BENCH_ROUTE")), logical_m(shape[0]),
          reduction(is_ag ? shape[1] : shape[1] / bootstrap.world),
          columns(is_ag ? shape[2] / bootstrap.world : shape[2]) {
        is_nt = route.rfind("ag_nt_", 0) == 0 || route == "rs_nt";
        std::stringstream partition(requiredEnvironment("FTRAIN_BENCH_RANK_M"));
        std::string token;
        while (std::getline(partition, token, ',')) { sizes.push_back(std::stoll(token)); }
        require(sizes.size() == size_t(bootstrap.world) &&
                    std::accumulate(sizes.begin(), sizes.end(), int64_t{0}) == logical_m,
                "invalid MoE benchmark rank partition");
        offsets.resize(bootstrap.world + 1, 0);
        std::partial_sum(sizes.begin(), sizes.end(), offsets.begin() + 1);
        routing_values.resize(size_t(logical_m) * topk);
        source_rows.resize(routing_values.size());
        const bool balanced = environmentInt("FTRAIN_BENCH_BALANCED", 0) != 0;
        for (size_t index = 0; index < routing_values.size(); ++index) {
            routing_values[index] = balanced ? index % experts : mixBits(uint32_t(index) + 3000U + salt) % experts;
        }
        if (is_ag) {
            const auto raw_ids = routing_values;
            for (int peer = 0; peer < bootstrap.world; ++peer) {
                std::vector<int64_t> order(size_t(sizes[peer]) * topk);
                std::iota(order.begin(), order.end(), offsets[peer] * topk);
                std::stable_sort(order.begin(), order.end(),
                                 [&](int64_t left, int64_t right) { return raw_ids[left] < raw_ids[right]; });
                for (size_t index = 0; index < order.size(); ++index) {
                    const auto destination      = offsets[peer] * topk + index;
                    routing_values[destination] = raw_ids[order[index]];
                    source_rows[destination]    = order[index] / topk;
                }
            }
        } else {
            std::iota(source_rows.begin(), source_rows.end(), 0);
        }
        const int input_rows = is_ag ? sizes[bootstrap.rank] * topk : logical_m * topk;
        tensors.push_back(
            std::make_unique<DeviceTensor>(std::vector<int64_t>{input_rows, reduction}, FTRAIN_NUMERIC_TYPE_BF16));
        tensors.push_back(std::make_unique<DeviceTensor>(is_nt ? std::vector<int64_t>{experts, columns, reduction}
                                                               : std::vector<int64_t>{experts, reduction, columns},
                                                         FTRAIN_NUMERIC_TYPE_BF16));
        tensors.push_back(std::make_unique<DeviceTensor>(
            std::vector<int64_t>{is_ag ? logical_m * topk : logical_m, is_ag ? 1 : topk}, FTRAIN_NUMERIC_TYPE_INT32));
        tensors.push_back(
            std::make_unique<DeviceTensor>(std::vector<int64_t>{logical_m, topk}, FTRAIN_NUMERIC_TYPE_BF16));
        tensors.push_back(std::make_unique<DeviceTensor>(
            std::vector<int64_t>{is_ag ? logical_m * topk : sizes[bootstrap.rank], columns}, FTRAIN_NUMERIC_TYPE_BF16));
        DeviceTensor device_rows({input_rows}, FTRAIN_NUMERIC_TYPE_INT32);
        const auto begin = source_rows.data() + (is_ag ? offsets[bootstrap.rank] * topk : 0);
        checkHip(
            hipMemcpy(device_rows.tensor.storage.memory, begin, input_rows * sizeof(int32_t), hipMemcpyHostToDevice),
            "copy input rows");
        checkHip(hipMemcpy(tensors[2]->tensor.storage.memory, routing_values.data(), tensors[2]->bytes,
                           hipMemcpyHostToDevice),
                 "copy routing");
        std::vector<uint16_t> probabilities(size_t(logical_m) * topk);
        const float denominator = float(topk * (topk + 1) / 2);
        for (size_t index = 0; index < probabilities.size(); ++index) {
            probabilities[index] = bf16(float(index % topk + 1) / denominator);
        }
        checkHip(hipMemcpy(tensors[3]->tensor.storage.memory, probabilities.data(), tensors[3]->bytes,
                           hipMemcpyHostToDevice),
                 "copy probabilities");
        fillInput<<<1024, 256>>>(tensors[0]->data<uint16_t>(), tensors[0]->elements, reduction,
                                 device_rows.data<int32_t>(), bootstrap.rank, salt);
        fillWeight<<<1024, 256>>>(tensors[1]->data<uint16_t>(), tensors[1]->elements, columns, reduction,
                                  bootstrap.rank, is_nt, salt);
        checkHip(hipGetLastError(), "fill MoE benchmark data");
        checkHip(hipDeviceSynchronize(), "data preparation");
        fingerprints = "[";
        for (size_t index = 0; index < 4; ++index) {
            fingerprints += (index ? "," : "") + tensors[index]->fingerprint();
        }
        fingerprints    += "]";
        auto rank_sizes  = sizes;
        if (is_ag) {
            for (auto& size : rank_sizes) { size *= topk; }
        }
        checkFtrain(ftrainCollectiveContextCreate(&context, bootstrap.rank, bootstrap.world, bootstrap.rank,
                                                  MoeContext::callback, &bootstrap),
                    "create MoE context");
        if (is_ag) {
            FTrainAGGroupGemmConfig config{};
            checkFtrain(ftrainAGGroupGemmConfigInit(&config), "AG config");
            config.m_dim            = logical_m * topk;
            config.n_dim            = columns;
            config.k_dim            = reduction;
            config.num_experts      = experts;
            config.topk             = 1;
            config.route            = route.c_str();
            config.rank_m_sizes     = rank_sizes.data();
            config.num_rank_m_sizes = rank_sizes.size();
            checkFtrain(ftrainAGGroupGemmCreate(&ag, context, &config, nullptr), "create AG benchmark session");
        } else {
            FTrainMoEReduceRSConfig config{};
            checkFtrain(ftrainMoEReduceRSConfigInit(&config), "RS config");
            config.m_dim            = logical_m;
            config.n_dim            = columns;
            config.k_dim            = reduction;
            config.num_experts      = experts;
            config.topk             = topk;
            config.route            = route.c_str();
            config.rank_m_sizes     = rank_sizes.data();
            config.num_rank_m_sizes = rank_sizes.size();
            config.n_chunks         = environmentInt("FTRAIN_BENCH_N_CHUNKS", 0);
            checkFtrain(ftrainMoEReduceRSCreate(&rs, context, &config, nullptr), "create RS benchmark session");
        }
        checkHip(hipDeviceSynchronize(), "MoE session preparation");
    }

    ~MoeSession() override {
        if (ag != nullptr) { ftrainAGGroupGemmDestroy(ag); }
        if (rs != nullptr) { ftrainMoEReduceRSDestroy(rs); }
        ftrainCollectiveContextDestroy(context);
    }

    int warmupIterations() const override { return environmentInt("FTRAIN_BENCH_WARMUP", 10); }
    int measurementIterations() const override { return environmentInt("FTRAIN_BENCH_ITERS", 40); }
    void barrier() override { bootstrap.barrier(); }
    std::string fingerprintJson() const override { return fingerprints; }

    void runConvenience(hipStream_t stream) override {
        if (is_ag) {
            checkFtrain(ftrainAGGroupGemmExecute(ag, tensors[0]->tensor, tensors[1]->tensor, tensors[2]->tensor,
                                                 tensors[4]->tensor, stream),
                        "AG execute");
        } else {
            checkFtrain(ftrainMoEReduceRSExecute(rs, tensors[0]->tensor, tensors[1]->tensor, tensors[2]->tensor,
                                                 tensors[3]->tensor, tensors[4]->tensor, stream),
                        "RS execute");
        }
    }

    void buildPattern(FTrainPattern pattern) override {
        ports.resize(tensors.size());
        for (size_t index = 0; index < ports.size(); ++index) {
            if (is_ag && index == 3) { continue; }
            checkFtrain(ftrainPatternAddTensor(pattern, &ports[index]), "MoE tensor port");
        }
        if (is_ag) {
            checkFtrain(ftrainPatternAddAGGroupGemm(pattern, &ag_op, ports[0], ports[1], ports[2], ports[4]),
                        "AG pattern");
        } else {
            checkFtrain(ftrainPatternAddMoEReduceRS(pattern, &rs_op, ports[0], ports[1], ports[2], ports[3], ports[4]),
                        "RS pattern");
        }
    }

    void bindArgs(FTrainArgs args) override {
        for (size_t index = 0; index < ports.size(); ++index) {
            if (is_ag && index == 3) { continue; }
            checkFtrain(ftrainArgsSetTensor(args, ports[index], tensors[index]->tensor.storage), "MoE tensor binding");
        }
        if (is_ag) {
            checkFtrain(ftrainArgsSetAGGroupGemm(args, ag_op, ag, 0, 0), "AG args");
        } else {
            checkFtrain(ftrainArgsSetMoEReduceRS(args, rs_op, rs, 0, 0), "RS args");
        }
    }

    float dot(int row, int expert, int column, int peer, int weight_rank) const {
        float sum = 0;
        for (int inner = 0; inner < reduction; ++inner) {
            sum += inputValue(row, inner, peer, salt) * weightValue(expert, column, inner, weight_rank, salt);
        }
        return fp32(bf16(sum));
    }

    void validate(hipStream_t stream) override {
        checkHip(hipMemsetAsync(tensors[4]->tensor.storage.memory, 0xff, tensors[4]->bytes, stream), "poison output");
        runConvenience(stream);
        validateOutput(stream);
    }

    void validateOutput(hipStream_t stream) override {
        int* invalid = nullptr;
        checkHip(hipMalloc(&invalid, sizeof(int)), "validation flag");
        checkHip(hipMemsetAsync(invalid, 0, sizeof(int), stream), "clear validation flag");
        inspect<<<1024, 256, 0, stream>>>(tensors[4]->data<uint16_t>(), tensors[4]->elements, invalid);
        checkHip(hipGetLastError(), "validate finite output");
        checkHip(hipStreamSynchronize(stream), "validate output sync");
        int nonfinite = 0;
        checkHip(hipMemcpy(&nonfinite, invalid, sizeof(int), hipMemcpyDeviceToHost), "read validation flag");
        checkHip(hipFree(invalid), "free validation flag");
        require(nonfinite == 0, "MoE output contains nonfinite values");
        for (uint64_t sample = 0; sample < 128; ++sample) {
            const auto index = sample * (tensors[4]->elements - 1) / 127;
            uint16_t value   = 0;
            checkHip(hipMemcpy(&value, tensors[4]->data<uint16_t>() + index, sizeof(value), hipMemcpyDeviceToHost),
                     "sample output");
            const int row    = index / columns;
            const int column = index % columns;
            float expected   = 0;
            if (is_ag) {
                const int owner = std::upper_bound(offsets.begin(), offsets.end(), row / topk) - offsets.begin() - 1;
                expected        = dot(source_rows[row], routing_values[row], column, owner, bootstrap.rank);
            } else {
                const int token = offsets[bootstrap.rank] + row;
                for (int peer = 0; peer < bootstrap.world; ++peer) {
                    float partial = 0;
                    for (int choice = 0; choice < topk; ++choice) {
                        const int expanded      = token * topk + choice;
                        const float probability = fp32(bf16(float(choice + 1) / float(topk * (topk + 1) / 2)));
                        partial +=
                            fp32(bf16(dot(expanded, routing_values[expanded], column, peer, peer) * probability));
                    }
                    expected += fp32(bf16(partial));
                }
                expected = fp32(bf16(expected));
            }
            if (std::abs(fp32(value) - expected) > 0.0005F + 0.03F * std::abs(expected)) {
                std::fprintf(stderr, "rank=%d sample=%llu row=%d column=%d actual=%.9g expected=%.9g inputs=%s\n",
                             bootstrap.rank, static_cast<unsigned long long>(sample), row, column, fp32(value),
                             expected, fingerprints.c_str());
                require(false, "MoE sampled output differs from CPU reference");
            }
        }
        barrier();
    }
};

struct MoeCase final : OpCase {
    bool is_ag;
    explicit MoeCase(bool ag) : is_ag(ag) {}
    void prepare() const override {
        requiredEnvironment("FTRAIN_BENCH_ROUTE");
        requiredEnvironment("FTRAIN_BENCH_SEED");
        requiredEnvironment("FTRAIN_BENCH_RANK_M");
        checkHip(hipSetDevice(environmentInt("FTRAIN_BENCH_RANK", 0)), "select rank device");
    }
    const char* name() const override { return is_ag ? "ag_group_gemm" : "moe_reduce_rs"; }
    const char* shapeGrammar() const override { return "global_m,global_k,global_n"; }
    const char* convenienceImpl() const override {
        return is_ag ? "ftrainAGGroupGemmExecute" : "ftrainMoEReduceRSExecute";
    }
    std::vector<std::pair<std::string, FTrainNumericType>> precisions() const override {
        return {
            {"bf16", FTRAIN_NUMERIC_TYPE_BF16}
        };
    }
    std::unique_ptr<OpSession> create(const std::vector<int64_t>& shape, FTrainNumericType) const override {
        require(shape.size() == 3, "MoE expects global M,K,N");
        return std::make_unique<MoeSession>(is_ag, shape);
    }
};

const bool registered = [] {
    registerOp(std::make_unique<MoeCase>(true));
    registerOp(std::make_unique<MoeCase>(false));
    return true;
}();

}  // namespace
}  // namespace ftrain_bench
