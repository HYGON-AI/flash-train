#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include <gtest/gtest.h>

#include "ag_default_environment.h"

namespace {

using ftrain::moe::core::ScopedAGDefaultEnvironment;

constexpr std::array<std::pair<const char*, const char*>, 7> kDefaults = {{
    {"FTRAIN_AG_GROUP_GEMM_ASM_MT256", "1"},
    {"FTRAIN_AG_GROUP_GEMM_RANK_TILE_ENABLE", "1"},
    {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_WGM8", "1"},
    {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", "0"},
    {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_VARIANT", "kn_direct"},
    {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT", "1"},
    {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", "1"},
}};

const char* failing_variable = nullptr;
int failed_setenv_calls = 0;

class MoeDefaultEnvironmentTest : public ::testing::Test {
  protected:
    void SetUp() override {
        failing_variable = nullptr;
        failed_setenv_calls = 0;
        for (std::size_t index = 0; index < kDefaults.size(); ++index) {
            if (const char* value = std::getenv(kDefaults[index].first)) { previous_values[index] = value; }
        }
        for (const auto& [name, value] : kDefaults) { ASSERT_EQ(unsetenv(name), 0); }
    }

    void TearDown() override {
        failing_variable = nullptr;
        for (std::size_t index = 0; index < kDefaults.size(); ++index) {
            if (previous_values[index]) {
                EXPECT_EQ(setenv(kDefaults[index].first, previous_values[index]->c_str(), 1), 0);
            } else {
                EXPECT_EQ(unsetenv(kDefaults[index].first), 0);
            }
        }
    }

    void expectDefaults() const {
        for (const auto& [name, value] : kDefaults) { EXPECT_STREQ(std::getenv(name), value) << name; }
    }

    void expectUnset() const {
        for (const auto& [name, value] : kDefaults) { EXPECT_EQ(std::getenv(name), nullptr) << name; }
    }

    std::array<std::optional<std::string>, kDefaults.size()> previous_values;
};

TEST_F(MoeDefaultEnvironmentTest, DisabledDoesNotSetDefaults) {
    failing_variable = kDefaults.front().first;
    ScopedAGDefaultEnvironment environment(false);
    expectUnset();
    EXPECT_EQ(failed_setenv_calls, 0);
}

TEST_F(MoeDefaultEnvironmentTest, AppliesAndRestoresDefaults) {
    {
        ScopedAGDefaultEnvironment environment(true);
        expectDefaults();
    }
    expectUnset();
}

TEST_F(MoeDefaultEnvironmentTest, PreservesUserSettings) {
    ASSERT_EQ(setenv(kDefaults[0].first, "ON", 1), 0);
    ASSERT_EQ(setenv(kDefaults[2].first, "0", 1), 0);
    ASSERT_EQ(setenv(kDefaults[4].first, "custom_variant", 1), 0);
    {
        ScopedAGDefaultEnvironment environment(true);
        EXPECT_STREQ(std::getenv(kDefaults[0].first), "ON");
        EXPECT_STREQ(std::getenv(kDefaults[2].first), "0");
        EXPECT_STREQ(std::getenv(kDefaults[4].first), "custom_variant");
        EXPECT_STREQ(std::getenv(kDefaults[1].first), kDefaults[1].second);
    }
    for (std::size_t index = 0; index < kDefaults.size(); ++index) {
        if (index == 0) { EXPECT_STREQ(std::getenv(kDefaults[index].first), "ON"); }
        else if (index == 2) { EXPECT_STREQ(std::getenv(kDefaults[index].first), "0"); }
        else if (index == 4) { EXPECT_STREQ(std::getenv(kDefaults[index].first), "custom_variant"); }
        else { EXPECT_EQ(std::getenv(kDefaults[index].first), nullptr); }
    }
}

TEST_F(MoeDefaultEnvironmentTest, DisabledMt256DoesNotSetDependentDefaults) {
    for (const char* value : {"0", "OFF", ""}) {
        ASSERT_EQ(setenv(kDefaults.front().first, value, 1), 0);
        {
            ScopedAGDefaultEnvironment environment(true);
            EXPECT_STREQ(std::getenv(kDefaults.front().first), value);
            for (std::size_t index = 1; index < kDefaults.size(); ++index) {
                EXPECT_EQ(std::getenv(kDefaults[index].first), nullptr);
            }
        }
        EXPECT_STREQ(std::getenv(kDefaults.front().first), value);
    }
}

TEST_F(MoeDefaultEnvironmentTest, NestedScopesPreserveOuterDefaults) {
    {
        ScopedAGDefaultEnvironment outer_environment(true);
        {
            ScopedAGDefaultEnvironment inner_environment(true);
            expectDefaults();
        }
        expectDefaults();
    }
    expectUnset();
}

TEST_F(MoeDefaultEnvironmentTest, FailedSetupPreservesUserSettings) {
    ASSERT_EQ(setenv(kDefaults[1].first, "0", 1), 0);
    failing_variable = kDefaults[2].first;
    EXPECT_THROW(ScopedAGDefaultEnvironment environment(true), std::system_error);
    EXPECT_STREQ(std::getenv(kDefaults[1].first), "0");
    for (std::size_t index = 0; index < kDefaults.size(); ++index) {
        if (index != 1) { EXPECT_EQ(std::getenv(kDefaults[index].first), nullptr); }
    }
    EXPECT_EQ(failed_setenv_calls, 1);
}

class MoeDefaultEnvironmentFailureTest : public MoeDefaultEnvironmentTest,
                                        public ::testing::WithParamInterface<std::size_t> {};

TEST_P(MoeDefaultEnvironmentFailureTest, ReportsFailureAndRollsBackBeforeRetry) {
    failing_variable = kDefaults[GetParam()].first;
    try {
        ScopedAGDefaultEnvironment environment(true);
        FAIL() << "setenv failure was ignored for " << failing_variable;
    } catch (const std::system_error& error) {
        EXPECT_EQ(error.code(), std::error_code(ENOMEM, std::generic_category()));
        EXPECT_NE(std::string(error.what()).find(failing_variable), std::string::npos);
    }
    EXPECT_EQ(failed_setenv_calls, 1);
    expectUnset();
    failing_variable = nullptr;
    {
        ScopedAGDefaultEnvironment environment(true);
        expectDefaults();
    }
    expectUnset();
}

INSTANTIATE_TEST_SUITE_P(AllDefaults, MoeDefaultEnvironmentFailureTest,
                         ::testing::Range<std::size_t>(0, kDefaults.size()));

}

extern "C" int __real_setenv(const char* name, const char* value, int overwrite) noexcept;

extern "C" int __wrap_setenv(const char* name, const char* value, int overwrite) noexcept {
    if (failing_variable != nullptr && std::strcmp(name, failing_variable) == 0) {
        ++failed_setenv_calls;
        errno = ENOMEM;
        return -1;
    }
    return __real_setenv(name, value, overwrite);
}
