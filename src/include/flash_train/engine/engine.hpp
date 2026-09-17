#ifndef FTRAIN_ENGINE_ENGINE_HPP_
#define FTRAIN_ENGINE_ENGINE_HPP_

// Convenience header for the engine layer: the OpsEngine interface plus
// every concrete ops engine. Modules that consume engines generically
// include this instead of each family header.

#include "flash_train/engine/base.hpp"
#include "flash_train/engine/gemm/gemm_engine.hpp"

#endif
