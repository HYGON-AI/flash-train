#include "flash_train/common.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "flash_train/api.hpp"
#include "flash_train/api_handles.hpp"
#include "flash_train/error.hpp"
#include "flash_train/op_definitions.hpp"
#include "flash_train/op_schema.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/trace.hpp"

namespace ftrain {
namespace {

void requirePattern(FTrainPattern pattern) {
    if (pattern == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "pattern must not be null"); }
}

template<typename Id>
void requireOutput(Id* output, const char* name) {
    if (output == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s must not be null", name); }
}

}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainPatternCreate(FTrainPattern* pattern) {
    ftrain::ScopedTraceRange trace_range{"ftrainPatternCreate"};
    return ftrain::invokeApi([&] {
        ftrain::requireOutput(pattern, "pattern");

        auto created_pattern = std::make_unique<FTrainPatternStruct>();
        *pattern             = created_pattern.release();
    });
}

extern "C" FTrainStatus ftrainPatternDestroy(FTrainPattern pattern) {
    ftrain::ScopedTraceRange trace_range{"ftrainPatternDestroy"};
    return ftrain::invokeApi([&] {
        ftrain::requirePattern(pattern);
        delete pattern;
    });
}

extern "C" FTrainStatus ftrainPatternAddTensor(FTrainPattern pattern, FTrainTensorId* tensor) {
    ftrain::ScopedTraceRange trace_range{"ftrainPatternAddTensor"};
    return ftrain::invokeApi([&] {
        ftrain::requirePattern(pattern);
        ftrain::requireOutput(tensor, "tensor");
        *tensor = ftrain::Operand<ftrain::OperandKind::kTensor>::addToPattern(pattern->pattern);
    });
}

extern "C" FTrainStatus ftrainPatternAddTensorList(FTrainPattern pattern, FTrainTensorListId* tensor_list) {
    ftrain::ScopedTraceRange trace_range{"ftrainPatternAddTensorList"};
    return ftrain::invokeApi([&] {
        ftrain::requirePattern(pattern);
        ftrain::requireOutput(tensor_list, "tensor_list");
        *tensor_list = ftrain::Operand<ftrain::OperandKind::kTensorList>::addToPattern(pattern->pattern);
    });
}

extern "C" FTrainStatus ftrainPatternAddGroupedTensor(FTrainPattern pattern, FTrainGroupedTensorId* grouped_tensor) {
    ftrain::ScopedTraceRange trace_range{"ftrainPatternAddGroupedTensor"};
    return ftrain::invokeApi([&] {
        ftrain::requirePattern(pattern);
        ftrain::requireOutput(grouped_tensor, "grouped_tensor");
        *grouped_tensor = ftrain::Operand<ftrain::OperandKind::kGroupedTensor>::addToPattern(pattern->pattern);
    });
}

extern "C" FTrainStatus ftrainPatternAddGroupedABCDGemm(FTrainPattern pattern, FTrainGroupedABCDGemmOpId* op,
                                                        FTrainGroupedTensorId a, FTrainGroupedTensorId b,
                                                        FTrainGroupedTensorId c, FTrainGroupedTensorId d,
                                                        FTrainTensorId alpha, FTrainTensorId beta) {
    ftrain::ScopedTraceRange trace_range{"ftrainPatternAddGroupedABCDGemm"};
    return ftrain::invokeApi([&] {
        ftrain::requirePattern(pattern);
        ftrain::requireOutput(op, "op");
        *op = ftrain::Operation<ftrain::OperationKind::kGroupedABCDGemm>::addToPattern(pattern->pattern, a, b, c, d,
                                                                                       alpha, beta);
    });
}

extern "C" FTrainStatus ftrainPatternAddGemm(FTrainPattern pattern, FTrainGemmOpId* op, FTrainTensorId a,
                                             FTrainTensorId b, FTrainTensorId c, FTrainTensorId d, FTrainTensorId alpha,
                                             FTrainTensorId beta) {
    ftrain::ScopedTraceRange trace_range{"ftrainPatternAddGemm"};
    return ftrain::invokeApi([&] {
        ftrain::requirePattern(pattern);
        ftrain::requireOutput(op, "op");
        *op = ftrain::Operation<ftrain::OperationKind::kGemm>::addToPattern(pattern->pattern, a, b, c, d, alpha, beta);
    });
}

extern "C" FTrainStatus ftrainPatternAddGroupedBCDGemm(FTrainPattern pattern, FTrainGroupedBCDGemmOpId* op,
                                                       FTrainTensorListId a, FTrainGroupedTensorId b,
                                                       FTrainGroupedTensorId c, FTrainGroupedTensorId d,
                                                       FTrainTensorId alpha, FTrainTensorId beta) {
    ftrain::ScopedTraceRange trace_range{"ftrainPatternAddGroupedBCDGemm"};
    return ftrain::invokeApi([&] {
        ftrain::requirePattern(pattern);
        ftrain::requireOutput(op, "op");
        *op = ftrain::Operation<ftrain::OperationKind::kGroupedBCDGemm>::addToPattern(pattern->pattern, a, b, c, d,
                                                                                      alpha, beta);
    });
}

extern "C" FTrainStatus ftrainPatternAddGroupedABGemm(FTrainPattern pattern, FTrainGroupedABGemmOpId* op,
                                                      FTrainGroupedTensorId a, FTrainGroupedTensorId b,
                                                      FTrainTensorListId c, FTrainTensorListId d, FTrainTensorId alpha,
                                                      FTrainTensorId beta) {
    ftrain::ScopedTraceRange trace_range{"ftrainPatternAddGroupedABGemm"};
    return ftrain::invokeApi([&] {
        ftrain::requirePattern(pattern);
        ftrain::requireOutput(op, "op");
        *op = ftrain::Operation<ftrain::OperationKind::kGroupedABGemm>::addToPattern(pattern->pattern, a, b, c, d,
                                                                                     alpha, beta);
    });
}
