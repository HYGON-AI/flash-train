// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// C harness：L2 C 便利 / L3 Plan 复用 / L4 逐 Primitive 三档口径。
// 形状由 Python 包装器从标准套件生成并经命令行传入，结果以 JSON 行打到 stdout；
// 与 Python 层共用同一计时段（事件计时、预热 10、自适应迭代、中位数主口径）。

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <flash_train/common.h>
#include <flash_train/flash_train.h>

namespace {

constexpr int kWarmup = 10;
constexpr double kTargetMs = 200.0;
constexpr int kMinIters = 10;
constexpr int kMaxIters = 200;

struct Shape {
    int64_t m, k, n;
};

struct Stats {
    double median_ms, mean_ms, min_ms, p10_ms, p90_ms;
    int iters;
};

void checkHip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "hip error at %s: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

void checkFtrain(FTrainStatus status, const char* what) {
    if (status != FTRAIN_STATUS_SUCCESS) {
        std::fprintf(stderr, "ftrain error at %s: status=%u msg=%s\n", what, status,
                     ftrainGetLastMessage());
        std::exit(1);
    }
}

template <typename Fn>
Stats measure(hipStream_t stream, Fn&& fn) {
    hipEvent_t start, stop;
    checkHip(hipEventCreateWithFlags(&start, hipEventDefault), "event create");
    checkHip(hipEventCreateWithFlags(&stop, hipEventDefault), "event create");

    for (int i = 0; i < kWarmup; ++i) fn();
    checkHip(hipStreamSynchronize(stream), "warmup sync");

    auto once = [&]() {
        checkHip(hipEventRecord(start, stream), "event record");
        fn();
        checkHip(hipEventRecord(stop, stream), "event record");
        checkHip(hipStreamSynchronize(stream), "iter sync");
        float ms = 0.0f;
        checkHip(hipEventElapsedTime(&ms, start, stop), "elapsed");
        return static_cast<double>(ms);
    };

    const double est = std::max(once(), 1e-3);
    const int iters =
        std::max(kMinIters, std::min(kMaxIters, static_cast<int>(kTargetMs / est) + 1));

    std::vector<double> samples;
    samples.reserve(iters);
    double sum = 0.0;
    for (int i = 0; i < iters; ++i) {
        const double ms = once();
        samples.push_back(ms);
        sum += ms;
    }
    std::sort(samples.begin(), samples.end());
    const int n = static_cast<int>(samples.size());

    Stats st{0.0, 0.0, 0.0, 0.0, 0.0, iters};
    st.median_ms = (n % 2 == 1) ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2.0;
    st.mean_ms = sum / n;
    st.min_ms = samples.front();
    st.p10_ms = samples[std::max(0, n / 10 - 1)];
    st.p90_ms = samples[std::min(n - 1, (n * 9) / 10)];

    hipEventDestroy(start);
    hipEventDestroy(stop);
    return st;
}

void printStats(const Stats& s) {
    std::printf(
        "{\"median_ms\":%.6f,\"mean_ms\":%.6f,\"min_ms\":%.6f,\"p10_ms\":%.6f,\"p90_ms\":%.6f,"
        "\"iters\":%d}",
        s.median_ms, s.mean_ms, s.min_ms, s.p10_ms, s.p90_ms, s.iters);
}

// 六端口视图：每个视图持有独立的 dims 数组；布局统一用 CONTINUOUS 预定义
// 布局（strides 为空），标量为秩 0，与库测试的规范构造一致。
struct GemmViews {
    std::vector<int64_t> a_dims, b_dims, cd_dims;
    float alpha_value = 1.0f;
    float beta_value = 0.0f;
    FTrainStorageView a{}, b{}, c{}, d{}, alpha{}, beta{};

    GemmViews(void* da, void* db, void* dd, const Shape& s)
        : a_dims{s.m, s.k}, b_dims{s.k, s.n}, cd_dims{s.m, s.n} {
        a = dense2d(da, a_dims.data());
        b = dense2d(db, b_dims.data());
        c = dense2d(nullptr, cd_dims.data());
        d = dense2d(dd, cd_dims.data());
        alpha = scalar(&alpha_value);
        beta = scalar(&beta_value);
    }

private:
    static FTrainStorageView dense2d(void* memory, const int64_t* dims) {
        FTrainStorageView v{};
        v.memory = memory;
        v.dims = dims;
        v.strides = nullptr;
        v.num_dims = 2;
        v.numeric_type = FTRAIN_NUMERIC_TYPE_FP32;
        v.index_type = FTRAIN_INDEX_TYPE_CONTINUOUS;
        v.is_host_memory = false;
        return v;
    }

    static FTrainStorageView scalar(float* value) {
        FTrainStorageView v{};
        v.memory = value;
        v.dims = nullptr;
        v.strides = nullptr;
        v.num_dims = 0;
        v.numeric_type = FTRAIN_NUMERIC_TYPE_FP32;
        v.index_type = FTRAIN_INDEX_TYPE_CONTINUOUS;
        v.is_host_memory = true;
        return v;
    }
};

class DeviceBuffers {
public:
    explicit DeviceBuffers(const Shape& shape) : shape_(shape) {
        const size_t a = static_cast<size_t>(shape.m) * shape.k;
        const size_t b = static_cast<size_t>(shape.k) * shape.n;
        const size_t d = static_cast<size_t>(shape.m) * shape.n;
        host_.resize(std::max({a, b, d}), 0.5f);
        checkHip(hipMalloc(&da_, a * sizeof(float)), "malloc a");
        checkHip(hipMalloc(&db_, b * sizeof(float)), "malloc b");
        checkHip(hipMalloc(&dd_, d * sizeof(float)), "malloc d");
        checkHip(hipMemcpy(da_, host_.data(), a * sizeof(float), hipMemcpyHostToDevice),
                 "fill a");
        checkHip(hipMemcpy(db_, host_.data(), b * sizeof(float), hipMemcpyHostToDevice),
                 "fill b");
        checkHip(hipMemcpy(dd_, host_.data(), d * sizeof(float), hipMemcpyHostToDevice),
                 "fill d");
    }

    ~DeviceBuffers() {
        hipFree(da_);
        hipFree(db_);
        hipFree(dd_);
    }

    GemmViews views() { return GemmViews(da_, db_, dd_, shape_); }

private:
    Shape shape_;
    std::vector<float> host_;
    void* da_ = nullptr;
    void* db_ = nullptr;
    void* dd_ = nullptr;
};

void runShape(const Shape& shape, hipStream_t stream, bool& first_row) {
    DeviceBuffers buf(shape);
    GemmViews v = buf.views();

    // L2：C 便利 API，每次调用全流程（对齐 Python 便利层口径）。
    {
        const Stats st = measure(stream, [&] {
            checkFtrain(ftrainGemm(v.a, v.b, v.c, v.d, v.alpha, v.beta,
                                   FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, stream),
                        "ftrainGemm");
        });
        if (!first_row) std::printf(",\n");
        first_row = false;
        std::printf(
            "{\"tier\":\"convenience-c\",\"impl\":\"ftrainGemm\",\"shape\":[%ld,%ld,%ld],"
            "\"stats\":",
            static_cast<long>(shape.m), static_cast<long>(shape.k),
            static_cast<long>(shape.n));
        printStats(st);
        std::printf("}");
    }

    // L3 / L4：分阶段 API，Plan 建好后按索引执行。
    FTrainPattern pattern;
    checkFtrain(ftrainPatternCreate(&pattern), "pattern create");
    FTrainTensorId ta, tb, tc, td, talpha, tbeta;
    checkFtrain(ftrainPatternAddTensor(pattern, &ta), "add a");
    checkFtrain(ftrainPatternAddTensor(pattern, &tb), "add b");
    checkFtrain(ftrainPatternAddTensor(pattern, &tc), "add c");
    checkFtrain(ftrainPatternAddTensor(pattern, &td), "add d");
    checkFtrain(ftrainPatternAddTensor(pattern, &talpha), "add alpha");
    checkFtrain(ftrainPatternAddTensor(pattern, &tbeta), "add beta");
    FTrainGemmOpId gemm;
    checkFtrain(ftrainPatternAddGemm(pattern, &gemm, ta, tb, tc, td, talpha, tbeta),
                "add gemm");

    FTrainOps ops;
    checkFtrain(ftrainOpsCreate(&ops, pattern), "ops create");
    ftrainPatternDestroy(pattern);

    FTrainArgs args;
    checkFtrain(ftrainArgsCreate(&args, ops), "args create");
    checkFtrain(ftrainArgsSetTensor(args, ta, v.a), "set a");
    checkFtrain(ftrainArgsSetTensor(args, tb, v.b), "set b");
    checkFtrain(ftrainArgsSetTensor(args, tc, v.c), "set c");
    checkFtrain(ftrainArgsSetTensor(args, td, v.d), "set d");
    checkFtrain(ftrainArgsSetTensor(args, talpha, v.alpha), "set alpha");
    checkFtrain(ftrainArgsSetTensor(args, tbeta, v.beta), "set beta");
    checkFtrain(ftrainArgsSetGemm(args, gemm, FTRAIN_NUMERIC_TYPE_FP32), "set gemm");

    FTrainPlan plan;
    checkFtrain(ftrainPlanCreate(&plan, args), "plan create");

    uint64_t num = 0;
    checkFtrain(ftrainPlanGetNumPrimitives(plan, &num), "num primitives");
    std::vector<const char*> names(num, nullptr);
    std::vector<uint64_t> ws_bytes(num, 0);
    uint64_t max_ws = 0;
    for (uint64_t i = 0; i < num; ++i) {
        checkFtrain(ftrainPlanGetPrimitiveName(plan, i, &names[i]), "primitive name");
        checkFtrain(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, i, &ws_bytes[i]),
                    "workspace bytes");
        max_ws = std::max(max_ws, ws_bytes[i]);
    }
    void* workspace = nullptr;
    if (max_ws > 0) checkHip(hipMalloc(&workspace, max_ws), "malloc workspace");

    // L3：推荐默认（primitive 0）的稳态执行。
    {
        const Stats st = measure(stream, [&] {
            checkFtrain(ftrainPlanExecutePrimitive(plan, 0, workspace, max_ws, stream),
                        "execute 0");
        });
        std::printf(",\n");
        std::printf(
            "{\"tier\":\"plan-reuse\",\"impl\":\"%s\",\"shape\":[%ld,%ld,%ld],"
            "\"workspace_bytes\":%llu,\"stats\":",
            names[0], static_cast<long>(shape.m), static_cast<long>(shape.k),
            static_cast<long>(shape.n), static_cast<unsigned long long>(ws_bytes[0]));
        printStats(st);
        std::printf("}");
    }

    // L4：Plan 内逐 Primitive，选择中立的实现矩阵。
    for (uint64_t i = 0; i < num; ++i) {
        const Stats st = measure(stream, [&] {
            checkFtrain(ftrainPlanExecutePrimitive(plan, i, workspace, max_ws, stream),
                        "execute i");
        });
        std::printf(",\n");
        std::printf(
            "{\"tier\":\"primitive\",\"impl\":\"%s\",\"index\":%llu,"
            "\"shape\":[%ld,%ld,%ld],\"workspace_bytes\":%llu,\"stats\":",
            names[i], static_cast<unsigned long long>(i), static_cast<long>(shape.m),
            static_cast<long>(shape.k), static_cast<long>(shape.n),
            static_cast<unsigned long long>(ws_bytes[i]));
        printStats(st);
        std::printf("}");
    }

    if (workspace != nullptr) hipFree(workspace);
    ftrainPlanDestroy(plan);
    ftrainArgsDestroy(args);
    ftrainOpsDestroy(ops);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <precision> <m,k,n> [<m,k,n> ...]\n", argv[0]);
        return 2;
    }
    const std::string precision = argv[1];
    if (precision != "fp32") {
        std::fprintf(stderr, "unsupported precision: %s\n", precision.c_str());
        return 2;
    }

    std::vector<Shape> shapes;
    for (int i = 2; i < argc; ++i) {
        Shape s{};
        if (std::sscanf(argv[i], "%ld,%ld,%ld", &s.m, &s.k, &s.n) != 3 || s.m <= 0 || s.k <= 0
            || s.n <= 0) {
            std::fprintf(stderr, "bad shape: %s (expected m,k,n)\n", argv[i]);
            return 2;
        }
        shapes.push_back(s);
    }

    // FTRAIN_BENCH_DEVICE 指定设备号（与 Python 侧同义）；基准计时对并发负载敏感。
    if (const char* dev_env = std::getenv("FTRAIN_BENCH_DEVICE")) {
        checkHip(hipSetDevice(std::atoi(dev_env)), "set device");
    }

    hipStream_t stream;
    checkHip(hipStreamCreate(&stream), "stream create");

    std::printf("{\"rows\":[\n");
    bool first_row = true;
    for (const Shape& shape : shapes) runShape(shape, stream, first_row);
    std::printf("]}\n");

    hipStreamDestroy(stream);
    return 0;
}
