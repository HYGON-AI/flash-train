// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// C harness 通用骨架：计时（与 Python 侧同口径）与三档测量（L2 C 便利 / L3 Plan
// 复用 / L4 逐 Primitive）对所有算子一致；算子差异全部收在 OpCase/OpSession
// 钩子里（见 op.hpp 与 ops/）。形状由 Python 包装器从标准套件生成并经命令行
// 传入，结果以 JSON 行打到 stdout。

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "op.hpp"

namespace ftrain_bench {
namespace {

std::vector<std::unique_ptr<OpCase>>& opRegistry() {
    static std::vector<std::unique_ptr<OpCase>> registry;
    return registry;
}

constexpr int kWarmup = 10;
constexpr double kTargetMs = 200.0;
constexpr int kMinIters = 10;
constexpr int kMaxIters = 200;

struct Stats {
    double median_ms, mean_ms, min_ms, p10_ms, p90_ms;
    int iters;
};

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

    checkHip(hipEventDestroy(start), "event destroy");
    checkHip(hipEventDestroy(stop), "event destroy");
    return st;
}

void printStats(const Stats& s) {
    std::printf(
        "{\"median_ms\":%.6f,\"mean_ms\":%.6f,\"min_ms\":%.6f,\"p10_ms\":%.6f,\"p90_ms\":%.6f,"
        "\"iters\":%d}",
        s.median_ms, s.mean_ms, s.min_ms, s.p10_ms, s.p90_ms, s.iters);
}

// extra_fields 可空，形如 "\"index\":1,\"workspace_bytes\":0"
void printRow(bool& first_row, const char* tier, const char* impl, const std::vector<int64_t>& shape,
              const char* extra_fields) {
    if (!first_row) std::printf(",\n");
    first_row = false;
    std::printf("{\"tier\":\"%s\",\"impl\":\"%s\",", tier, impl);
    if (extra_fields != nullptr) std::printf("%s,", extra_fields);
    std::printf("\"shape\":[");
    for (size_t i = 0; i < shape.size(); ++i) {
        std::printf("%s%ld", i == 0 ? "" : ",", static_cast<long>(shape[i]));
    }
    std::printf("],\"stats\":");
}

void runShape(const OpCase& op, FTrainNumericType compute, const std::vector<int64_t>& shape,
              hipStream_t stream, bool& first_row) {
    std::unique_ptr<OpSession> session = op.create(shape, compute);

    // L2：便利 API，每次调用全流程（对齐 Python 便利层口径）
    {
        const Stats st = measure(stream, [&] { session->runConvenience(stream); });
        printRow(first_row, "convenience-c", op.convenienceImpl(), shape, nullptr);
        printStats(st);
        std::printf("}");
    }

    // L3 / L4：分阶段 API，Plan 建好后按索引执行
    FTrainPattern pattern = nullptr;
    checkFtrain(ftrainPatternCreate(&pattern), "pattern create");
    session->buildPattern(pattern);
    FTrainOps ops = nullptr;
    checkFtrain(ftrainOpsCreate(&ops, pattern), "ops create");
    ftrainPatternDestroy(pattern);

    FTrainArgs args = nullptr;
    checkFtrain(ftrainArgsCreate(&args, ops), "args create");
    session->bindArgs(args);

    FTrainPlan plan = nullptr;
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

    // L3：推荐默认（primitive 0）的稳态执行
    {
        const Stats st = measure(stream, [&] {
            checkFtrain(ftrainPlanExecutePrimitive(plan, 0, workspace, max_ws, stream),
                        "execute 0");
        });
        char extra[64];
        std::snprintf(extra, sizeof(extra), "\"workspace_bytes\":%llu",
                      static_cast<unsigned long long>(ws_bytes[0]));
        printRow(first_row, "plan-reuse", names[0], shape, extra);
        printStats(st);
        std::printf("}");
    }

    // L4：Plan 内逐 Primitive，选择中立的实现矩阵
    for (uint64_t i = 0; i < num; ++i) {
        const Stats st = measure(stream, [&] {
            checkFtrain(ftrainPlanExecutePrimitive(plan, i, workspace, max_ws, stream),
                        "execute i");
        });
        char extra[96];
        std::snprintf(extra, sizeof(extra), "\"index\":%llu,\"workspace_bytes\":%llu",
                      static_cast<unsigned long long>(i),
                      static_cast<unsigned long long>(ws_bytes[i]));
        printRow(first_row, "primitive", names[i], shape, extra);
        printStats(st);
        std::printf("}");
    }

    if (workspace != nullptr) hipFree(workspace);
    ftrainPlanDestroy(plan);
    ftrainArgsDestroy(args);
    ftrainOpsDestroy(ops);
}

std::vector<int64_t> parseShape(const char* text) {
    const std::string s(text);
    std::vector<int64_t> dims;
    size_t pos = 0;
    while (true) {
        const size_t comma = s.find(',', pos);
        const std::string token = s.substr(pos, comma == std::string::npos ? std::string::npos
                                                                            : comma - pos);
        char* end = nullptr;
        const long long value = std::strtoll(token.c_str(), &end, 10);
        require(!token.empty() && end != nullptr && *end == '\0' && value > 0,
                "bad shape: expected positive comma-separated integers");
        dims.push_back(value);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return dims;
}

}  // namespace

void registerOp(std::unique_ptr<OpCase> op_case) {
    opRegistry().push_back(std::move(op_case));
}

const OpCase* findOp(const std::string& name) {
    for (const auto& entry : opRegistry()) {
        if (name == entry->name()) return entry.get();
    }
    return nullptr;
}

std::string registeredOpNames() {
    std::string names;
    for (const auto& entry : opRegistry()) {
        if (!names.empty()) names += ", ";
        names += entry->name();
    }
    return names;
}

}  // namespace ftrain_bench

int main(int argc, char** argv) {
    using namespace ftrain_bench;

    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <op> <precision> <m,k,n> [<m,k,n> ...]\nops: %s\n", argv[0],
                     registeredOpNames().c_str());
        return 2;
    }
    const std::string op_name = argv[1];
    const OpCase* op = findOp(op_name);
    if (op == nullptr) {
        std::fprintf(stderr, "unknown op: %s (available: %s)\n", op_name.c_str(),
                     registeredOpNames().c_str());
        return 2;
    }

    const std::string precision = argv[2];
    FTrainNumericType compute{};
    bool supported = false;
    for (const auto& [name, type] : op->precisions()) {
        if (name == precision) {
            compute = type;
            supported = true;
            break;
        }
    }
    require(supported, "unsupported precision for this op");

    std::vector<std::vector<int64_t>> shapes;
    for (int i = 3; i < argc; ++i) shapes.push_back(parseShape(argv[i]));

    hipStream_t stream;
    checkHip(hipStreamCreate(&stream), "stream create");

    std::printf("{\"rows\":[\n");
    bool first_row = true;
    for (const auto& shape : shapes) runShape(*op, compute, shape, stream, first_row);
    std::printf("]}\n");

    hipStreamDestroy(stream);
    return 0;
}
