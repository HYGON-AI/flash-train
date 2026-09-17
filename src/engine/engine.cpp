#include "flash_train/engine/engine.hpp"

namespace ftrain {

// The built-in engine catalog and the single place that enumerates the
// concrete operator families. Engine developers add new families under
// engine/<family>/ and register them by adding one factory here plus one
// include in engine.hpp; the process registry picks the whole list up
// through makeBuiltinOpsEngines.
std::vector<std::shared_ptr<OpsEngineBase>> makeBuiltinOpsEngines() { return {makeGemmOpsEngine()}; }

}  // namespace ftrain
