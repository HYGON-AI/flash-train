#ifndef FTRAIN_ENGINE_ENGINE_HPP_
#define FTRAIN_ENGINE_ENGINE_HPP_

// Convenience header for the engine layer: the OpsEngine interface plus
// every concrete ops engine. Modules that consume engines generically
// include this instead of each family header.

#include <memory>
#include <vector>

#include "flash_train/engine/base.hpp"
#include "flash_train/engine/gemm/gemm_engine.hpp"

namespace ftrain {

// Returns every built-in engine. The process registry registers this
// whole list once; an engine family joins the library by adding its
// factory to this list and its header above, and nowhere else.
// Allocation failure throws std::bad_alloc.
std::vector<std::shared_ptr<OpsEngineBase>> makeBuiltinOpsEngines();

}  // namespace ftrain

#endif
