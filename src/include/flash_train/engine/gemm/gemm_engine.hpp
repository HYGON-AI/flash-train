#ifndef FTRAIN_ENGINE_GEMM_GEMM_ENGINE_HPP_
#define FTRAIN_ENGINE_GEMM_GEMM_ENGINE_HPP_

#include <memory>

#include "flash_train/engine/base.hpp"
#include "flash_train/ops/gemm/problem.hpp"

namespace ftrain {

// Creates the built-in Gemm engine: six Tensor operands (a, b, c, alpha,
// beta, d) and one Gemm operation. Allocation failure throws std::bad_alloc.
std::shared_ptr<OpsEngineBase> makeGemmOpsEngine();

}  // namespace ftrain

#endif
