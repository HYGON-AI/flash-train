#include <cstdio>

#include "flash_train/error.hpp"

namespace ftrain {

void Result::set(FTrainStatus status, const char* message) noexcept {
    status_     = status;
    message_[0] = '\0';

    if (status == FTRAIN_STATUS_SUCCESS || message == nullptr) { return; }

    std::size_t index = 0;
    while (index + 1 < message_.size() && message[index] != '\0') {
        message_[index] = message[index];
        ++index;
    }
    message_[index] = '\0';
}

void Result::setFormat(FTrainStatus status, const char* format, ...) noexcept {
    std::va_list arguments;
    va_start(arguments, format);
    setVFormat(status, format, arguments);
    va_end(arguments);
}

void Result::setVFormat(FTrainStatus status, const char* format, std::va_list arguments) noexcept {
    status_     = status;
    message_[0] = '\0';

    if (status == FTRAIN_STATUS_SUCCESS || format == nullptr) { return; }

    const int format_result = std::vsnprintf(message_.data(), message_.size(), format, arguments);
    if (format_result < 0) {
        message_[0] = '\0';
        return;
    }
    message_.back() = '\0';
}

Exception::Exception(const Result& result) noexcept : result_(result) {
    if (result_.isSuccess()) {
        result_.set(FTRAIN_STATUS_INTERNAL_ERROR, "Exception cannot be constructed with FTRAIN_STATUS_SUCCESS");
    }
}

Exception::Exception(FTrainStatus status, const char* format, ...) noexcept {
    if (status == FTRAIN_STATUS_SUCCESS) {
        result_.set(FTRAIN_STATUS_INTERNAL_ERROR, "Exception cannot be constructed with FTRAIN_STATUS_SUCCESS");
        return;
    }

    std::va_list arguments;
    va_start(arguments, format);
    result_.setVFormat(status, format, arguments);
    va_end(arguments);
}

}  // namespace ftrain
