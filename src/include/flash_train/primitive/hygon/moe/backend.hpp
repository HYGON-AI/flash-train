#ifndef FTRAIN_PRIMITIVE_HYGON_MOE_BACKEND_HPP_
#define FTRAIN_PRIMITIVE_HYGON_MOE_BACKEND_HPP_

#include <memory>

#include "flash_train/family/moe/session.hpp"

namespace ftrain {

bool isMoeBackendEnabled() noexcept;
std::unique_ptr<MoeBackend> makeMoeBackend(std::shared_ptr<CollectiveContext> context,
                                        const MoeConfig& config, FTrainStream stream);

}

#endif
