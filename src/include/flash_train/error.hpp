// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_ERROR_HPP_
#define FTRAIN_ERROR_HPP_

#include <array>
#include <cstdarg>
#include <cstddef>
#include <exception>

#include "flash_train/common.h"

namespace ftrain {

// A status code plus a diagnostic message stored inside the object (at most
// 2047 characters). A successful Result always has an empty message; the
// message is always null-terminated. Copies are independent.
struct Result final {
    Result() noexcept = default;

    // Initializes with the same status and plain-text message semantics as set().
    Result(FTrainStatus status, const char* message) noexcept { set(status, message); }

    void clear() noexcept {
        status_     = FTRAIN_STATUS_SUCCESS;
        message_[0] = '\0';
    }

    // Stores status and copies message as plain text (percent signs are
    // literal). A success status or null message stores an empty message;
    // otherwise message must be null-terminated. Text longer than the capacity
    // is truncated.
    void set(FTrainStatus status, const char* message) noexcept;

    // Stores status and formats the message printf-style; format must be a
    // valid format string for the supplied arguments. A success status or null
    // format stores an empty message. Oversized output is truncated; a
    // formatting failure keeps the status and stores an empty message.
    void setFormat(FTrainStatus status, const char* format, ...) noexcept;

    bool isSuccess() const noexcept { return status_ == FTRAIN_STATUS_SUCCESS; }

    FTrainStatus getStatus() const noexcept { return status_; }

    // Returns this Result's message: always non-null and null-terminated. The
    // pointer stays valid until this Result is modified or destroyed.
    const char* getMessage() const noexcept { return message_.data(); }

  private:
    // Total message storage capacity, including the terminating null character.
    static constexpr std::size_t kMessageCapacity = 2048;
    friend class Exception;

    void setVFormat(FTrainStatus status, const char* format, std::va_list arguments) noexcept;

    FTrainStatus status_{FTRAIN_STATUS_SUCCESS};
    std::array<char, kMessageCapacity> message_{};
};

// An exception carrying a non-success Result; what() returns that Result's
// message. Constructing an Exception with FTRAIN_STATUS_SUCCESS instead stores
// FTRAIN_STATUS_INTERNAL_ERROR with the message "Exception cannot be
// constructed with FTRAIN_STATUS_SUCCESS".
class Exception final : public std::exception {
  public:
    // Stores a copy of result.
    explicit Exception(const Result& result) noexcept;

    // Stores status and a printf-formatted message.
    Exception(FTrainStatus status, const char* format, ...) noexcept;

    // Returns the stored Result; the reference stays valid for this
    // exception's lifetime.
    const Result& getResult() const noexcept { return result_; }

    const char* what() const noexcept override { return result_.getMessage(); }

  private:
    Result result_;
};

}  // namespace ftrain

#endif
