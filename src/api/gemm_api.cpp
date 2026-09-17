#include "flash_train/flash_train.h"

#include "flash_train/trace.hpp"
#include "flash_train/api.hpp"

namespace ftrain {
namespace {

// The Gemm convenience path's immutable pieces, built once per process
// through the staged C API: one Ops for the supported Gemm Pattern and the
// user-side role IDs for binding ports. Every call creates one Args, one
// Plan, and executes the recommended Primitive through the same API.
struct GemmEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId a{};
    FTrainTensorId b{};
    FTrainTensorId c{};
    FTrainTensorId d{};
    FTrainTensorId alpha{};
    FTrainTensorId beta{};
    FTrainGemmOpId gemm{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const GemmEntry& gemmEntry() {
    static const GemmEntry entry = [] {
        GemmEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.a); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.b); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.c); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.alpha); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.beta); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.d); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status =
                ftrainPatternAddGemm(pattern, &entry.gemm, entry.a, entry.b, entry.c, entry.d, entry.alpha, entry.beta);
        }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainOpsCreate(&entry.ops, pattern); }
        if (pattern != nullptr) { ftrainPatternDestroy(pattern); }
        entry.status = status;
        return entry;
    }();
    return entry;
}

}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainGemm(FTrainStorageView a, FTrainStorageView b, FTrainStorageView c, FTrainStorageView d,
                                   FTrainStorageView alpha, FTrainStorageView beta, FTrainNumericType compute_type,
                                   void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainGemm"};
    const ftrain::GemmEntry& entry = ftrain::gemmEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }

    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.a, a); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.b, b); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.c, c); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.d, d); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.alpha, alpha); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.beta, beta); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetGemm(args, entry.gemm, compute_type); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPlanCreate(&plan, args); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainPlanExecutePrimitive(plan, 0, workspace, workspace_bytes, stream);
    }

    // Keep the failing call's diagnostic: destroying handles on the way out
    // would otherwise reset the thread's last result.
    const ftrain::Result failure = status != FTRAIN_STATUS_SUCCESS ? ftrain::getLastResult() : ftrain::Result{};
    if (plan != nullptr) { ftrainPlanDestroy(plan); }
    if (args != nullptr) { ftrainArgsDestroy(args); }
    if (status != FTRAIN_STATUS_SUCCESS) { ftrain::setLastResult(failure.getStatus(), failure.getMessage()); }
    return status;
}
