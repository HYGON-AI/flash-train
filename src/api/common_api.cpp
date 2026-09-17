#include "flash_train/common.h"

#include "flash_train/api.hpp"

namespace ftrain {
namespace {

Result& getMutableLastResult() noexcept {
    static thread_local Result tls_last_result;
    return tls_last_result;
}

}  // namespace

const Result& getLastResult() noexcept { return getMutableLastResult(); }

void clearLastResult() noexcept { getMutableLastResult().clear(); }

void setLastResult(FTrainStatus status, const char* message) noexcept { getMutableLastResult().set(status, message); }

}  // namespace ftrain

extern "C" FTrainStatus ftrainGetLastStatus(void) { return ftrain::getLastResult().getStatus(); }

extern "C" const char* ftrainGetLastMessage(void) { return ftrain::getLastResult().getMessage(); }
