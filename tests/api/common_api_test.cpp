#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/api.hpp"

namespace ftrain {
namespace {

using VoidFunction = void (*)();

static_assert(noexcept(invokeApi(std::declval<VoidFunction>())));
static_assert(std::is_same_v<decltype(invokeApi(std::declval<VoidFunction>())), FTrainStatus>);

TEST(LastResultTest, FreshThreadStartsWithSuccess) {
    FTrainStatus status = FTRAIN_STATUS_INTERNAL_ERROR;
    std::string message = "not initialized";

    std::thread thread([&] {
        status  = ftrainGetLastStatus();
        message = ftrainGetLastMessage();
    });
    thread.join();

    EXPECT_EQ(status, FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(message, "");
}

TEST(LastResultTest, SetOverwriteAndClearCurrentThread) {
    clearLastResult();

    setLastResult(FTRAIN_STATUS_INVALID_ARGUMENT, "first error");
    EXPECT_EQ(getLastResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(getLastResult().getMessage(), "first error");

    setLastResult(FTRAIN_STATUS_UNSUPPORTED, "replacement error");
    EXPECT_EQ(getLastResult().getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_STREQ(getLastResult().getMessage(), "replacement error");

    clearLastResult();
    EXPECT_TRUE(getLastResult().isSuccess());
    EXPECT_STREQ(getLastResult().getMessage(), "");
}

TEST(LastResultTest, IsIsolatedBetweenThreads) {
    clearLastResult();
    setLastResult(FTRAIN_STATUS_INVALID_ARGUMENT, "main thread error");

    FTrainStatus child_initial_status = FTRAIN_STATUS_INTERNAL_ERROR;
    std::string child_initial_message;
    FTrainStatus child_final_status = FTRAIN_STATUS_INTERNAL_ERROR;
    std::string child_final_message;

    std::thread thread([&] {
        child_initial_status  = ftrainGetLastStatus();
        child_initial_message = ftrainGetLastMessage();

        setLastResult(FTRAIN_STATUS_UNSUPPORTED, "child thread error");
        child_final_status  = ftrainGetLastStatus();
        child_final_message = ftrainGetLastMessage();
    });
    thread.join();

    EXPECT_EQ(child_initial_status, FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(child_initial_message, "");
    EXPECT_EQ(child_final_status, FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(child_final_message, "child thread error");
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(ftrainGetLastMessage(), "main thread error");

    clearLastResult();
}

TEST(InvokeApiTest, SuccessClearsPreviousError) {
    setLastResult(FTRAIN_STATUS_INTERNAL_ERROR, "old error");
    bool callable_invoked = false;

    const FTrainStatus status = invokeApi([&] { callable_invoked = true; });

    EXPECT_TRUE(callable_invoked);
    EXPECT_EQ(status, FTRAIN_STATUS_SUCCESS);
    EXPECT_TRUE(getLastResult().isSuccess());
    EXPECT_STREQ(getLastResult().getMessage(), "");
}

TEST(InvokeApiTest, PreservesFTrainException) {
    const FTrainStatus status =
        invokeApi([] { throw Exception(FTRAIN_STATUS_UNSUPPORTED, "unsupported ops %d", 7); });

    EXPECT_EQ(status, FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(getLastResult().getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_STREQ(getLastResult().getMessage(), "unsupported ops 7");

    clearLastResult();
}

TEST(InvokeApiTest, MapsBadAllocToOutOfMemory) {
    const std::bad_alloc expected_exception;
    const std::string expected_message = expected_exception.what();

    const FTrainStatus status = invokeApi([&] { throw expected_exception; });

    EXPECT_EQ(status, FTRAIN_STATUS_OUT_OF_MEMORY);
    EXPECT_EQ(getLastResult().getStatus(), FTRAIN_STATUS_OUT_OF_MEMORY);
    EXPECT_EQ(getLastResult().getMessage(), expected_message);

    clearLastResult();
}

TEST(InvokeApiTest, MapsStandardExceptionToInternalError) {
    const FTrainStatus status = invokeApi([] { throw std::runtime_error("standard exception"); });

    EXPECT_EQ(status, FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(getLastResult().getStatus(), FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(getLastResult().getMessage(), "standard exception");

    clearLastResult();
}

TEST(InvokeApiTest, MapsUnknownExceptionToInternalError) {
    const FTrainStatus status = invokeApi([] { throw 42; });

    EXPECT_EQ(status, FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(getLastResult().getStatus(), FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(getLastResult().getMessage(), "Unknown exception caught at C API boundary");

    clearLastResult();
}

TEST(CommonApiTest, QueryFunctionsPreserveLastResultAndMessagePointer) {
    setLastResult(FTRAIN_STATUS_OVERFLOW, "overflow diagnostic");

    const char* message = ftrainGetLastMessage();

    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_OVERFLOW);
    EXPECT_EQ(ftrainGetLastMessage(), message);
    EXPECT_STREQ(message, "overflow diagnostic");
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_OVERFLOW);

    clearLastResult();
}

}  // namespace
}  // namespace ftrain
