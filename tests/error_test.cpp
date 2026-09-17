#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "flash_train/error.hpp"

namespace ftrain {
namespace {

static_assert(std::is_nothrow_default_constructible_v<Result>);
static_assert(std::is_nothrow_copy_constructible_v<Result>);
static_assert(std::is_nothrow_copy_assignable_v<Result>);
static_assert(std::is_nothrow_constructible_v<Result, FTrainStatus, const char*>);
static_assert(noexcept(std::declval<Result&>().clear()));
static_assert(noexcept(std::declval<Result&>().set(FTRAIN_STATUS_INTERNAL_ERROR, "error")));
static_assert(noexcept(std::declval<Result&>().setFormat(FTRAIN_STATUS_INTERNAL_ERROR, "%d", 1)));
static_assert(noexcept(std::declval<const Result&>().isSuccess()));
static_assert(noexcept(std::declval<const Result&>().getStatus()));
static_assert(noexcept(std::declval<const Result&>().getMessage()));
static_assert(std::is_nothrow_constructible_v<Exception, const Result&>);
static_assert(std::is_nothrow_constructible_v<Exception, FTrainStatus, const char*>);
static_assert(std::is_nothrow_copy_constructible_v<Exception>);

// Mirrors Result's private message capacity so the boundary tests can assert
// the externally observable preservation and truncation limits.
constexpr std::size_t kMessageCapacity = 2048;

TEST(ResultTest, DefaultConstructionRepresentsSuccess) {
    const Result result;

    EXPECT_TRUE(result.isSuccess());
    EXPECT_EQ(result.getStatus(), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(result.getMessage(), "");
}

TEST(ResultTest, PlainMessageIsCopiedWithoutFormatting) {
    Result result(FTRAIN_STATUS_INVALID_ARGUMENT, "progress is 100%");

    EXPECT_FALSE(result.isSuccess());
    EXPECT_EQ(result.getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(result.getMessage(), "progress is 100%");
}

TEST(ResultTest, SetOverwritesExistingErrorInPlace) {
    Result result(FTRAIN_STATUS_INTERNAL_ERROR, "a much longer previous error message");

    result.set(FTRAIN_STATUS_UNSUPPORTED, "short");

    EXPECT_EQ(result.getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_STREQ(result.getMessage(), "short");
}

TEST(ResultTest, NullMessageProducesEmptyMessage) {
    Result result(FTRAIN_STATUS_OUT_OF_MEMORY, nullptr);

    EXPECT_EQ(result.getStatus(), FTRAIN_STATUS_OUT_OF_MEMORY);
    EXPECT_STREQ(result.getMessage(), "");
}

TEST(ResultTest, SuccessStatusAlwaysClearsMessage) {
    Result result(FTRAIN_STATUS_INTERNAL_ERROR, "old error");

    result.set(FTRAIN_STATUS_SUCCESS, "ignored message");

    EXPECT_TRUE(result.isSuccess());
    EXPECT_STREQ(result.getMessage(), "");
}

TEST(ResultTest, ClearRestoresSuccessAndEmptyMessage) {
    Result result(FTRAIN_STATUS_INTERNAL_ERROR, "old error");

    result.clear();

    EXPECT_TRUE(result.isSuccess());
    EXPECT_EQ(result.getStatus(), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(result.getMessage(), "");
}

TEST(ResultTest, MessageAtCapacityBoundaryIsPreserved) {
    const std::string message(kMessageCapacity - 1, 'a');
    Result result(FTRAIN_STATUS_INTERNAL_ERROR, message.c_str());

    EXPECT_EQ(std::strlen(result.getMessage()), kMessageCapacity - 1);
    EXPECT_EQ(result.getMessage()[kMessageCapacity - 1], '\0');
    EXPECT_EQ(result.getMessage(), message);
}

TEST(ResultTest, OversizedPlainMessageIsTruncatedAndTerminated) {
    const std::string message(kMessageCapacity + 64, 'b');
    Result result(FTRAIN_STATUS_INTERNAL_ERROR, message.c_str());

    EXPECT_EQ(std::strlen(result.getMessage()), kMessageCapacity - 1);
    EXPECT_EQ(result.getMessage()[kMessageCapacity - 1], '\0');
    EXPECT_EQ(std::string(result.getMessage()), message.substr(0, kMessageCapacity - 1));
}

TEST(ResultTest, SetFormatFormatsAndStoresMessage) {
    Result result;

    result.setFormat(FTRAIN_STATUS_INVALID_ARGUMENT, "op=%s, group=%u, size=%lld", "gemm", 3U, 4097LL);

    EXPECT_EQ(result.getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(result.getMessage(), "op=gemm, group=3, size=4097");
}

TEST(ResultTest, SetFormatTruncatesOversizedMessage) {
    const std::string message(kMessageCapacity + 64, 'c');
    Result result;

    result.setFormat(FTRAIN_STATUS_INTERNAL_ERROR, "%s", message.c_str());

    EXPECT_EQ(std::strlen(result.getMessage()), kMessageCapacity - 1);
    EXPECT_EQ(result.getMessage()[kMessageCapacity - 1], '\0');
    EXPECT_EQ(std::string(result.getMessage()), message.substr(0, kMessageCapacity - 1));
}

TEST(ResultTest, NullFormatProducesEmptyMessage) {
    Result result;

    result.setFormat(FTRAIN_STATUS_INTERNAL_ERROR, nullptr);

    EXPECT_EQ(result.getStatus(), FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(result.getMessage(), "");
}

TEST(ResultTest, SuccessSetFormatDoesNotEvaluateFormat) {
    Result result(FTRAIN_STATUS_INTERNAL_ERROR, "old error");

    result.setFormat(FTRAIN_STATUS_SUCCESS, "%d");

    EXPECT_TRUE(result.isSuccess());
    EXPECT_STREQ(result.getMessage(), "");
}

TEST(ResultTest, CopyOwnsIndependentMessage) {
    Result original(FTRAIN_STATUS_INVALID_ARGUMENT, "original error");
    Result copy = original;

    original.set(FTRAIN_STATUS_UNSUPPORTED, "replacement error");

    EXPECT_EQ(copy.getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(copy.getMessage(), "original error");
}

TEST(ExceptionTest, CopiesResultAndOwnsItsMessage) {
    Result source(FTRAIN_STATUS_UNSUPPORTED, "stored failure");
    Exception exception(source);

    source.set(FTRAIN_STATUS_INTERNAL_ERROR, "replacement failure");

    EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_STREQ(exception.getResult().getMessage(), "stored failure");
    EXPECT_EQ(exception.what(), exception.getResult().getMessage());
}

TEST(ExceptionTest, FormatsDiagnosticMessage) {
    Exception exception(FTRAIN_STATUS_INVALID_ARGUMENT, "op=%s, argument=%d", "gemm", 7);

    EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(exception.what(), "op=gemm, argument=7");
}

TEST(ExceptionTest, ConvertsSuccessfulResultToInternalError) {
    const Result success;
    const Exception exception(success);

    EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_FALSE(exception.getResult().isSuccess());
    EXPECT_STREQ(exception.what(), "Exception cannot be constructed with FTRAIN_STATUS_SUCCESS");
}

TEST(ExceptionTest, ConvertsSuccessfulStatusToInternalError) {
    const Exception exception(FTRAIN_STATUS_SUCCESS, "ignored message");

    EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_FALSE(exception.getResult().isSuccess());
    EXPECT_STREQ(exception.what(), "Exception cannot be constructed with FTRAIN_STATUS_SUCCESS");
}

TEST(ExceptionTest, IsCatchableAsStandardException) {
    try {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "diagnostic %d", 42);
    }
    catch (const std::exception& exception) {
        EXPECT_STREQ(exception.what(), "diagnostic 42");
        return;
    }
    FAIL() << "Exception was not caught as std::exception";
}

}  // namespace
}  // namespace ftrain
