// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <hygon_vendor.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <utility>

namespace {

class RestoreEnvironment final {
  public:
    explicit RestoreEnvironment(std::string name)
        : name_(std::move(name)), old_value_(std::getenv(name_.c_str())) {}

    ~RestoreEnvironment() {
        if (old_value_ == nullptr) {
            unsetenv(name_.c_str());
        } else {
            setenv(name_.c_str(), old_value_, 1);
        }
    }

  private:
    std::string name_;
    const char *old_value_;
};

constexpr const char *kPrimaryName = "FTRAIN_HYGON_VENDOR_ENV_TEST";
constexpr const char *kLegacyName = "FL" "UX_VENDOR_ENV_TEST";

}  // namespace

TEST(HygonVendorEnvTest, HygonNameIsUsedWhenLegacyNameIsAbsent) {
    using bytedance::hygon_vendor::detail::get_vendor_bool_env;
    using bytedance::hygon_vendor::detail::get_vendor_env;

    RestoreEnvironment legacy(kLegacyName);
    RestoreEnvironment primary(kPrimaryName);
    unsetenv(kLegacyName);
    setenv(kPrimaryName, "ON", 1);

    EXPECT_EQ(std::string(get_vendor_env(kPrimaryName)), "ON");
    EXPECT_TRUE(get_vendor_bool_env(kPrimaryName, false));
}

TEST(HygonVendorEnvTest, PrimaryNameWinsDuringMigration) {
    using bytedance::hygon_vendor::detail::get_vendor_bool_env;
    using bytedance::hygon_vendor::detail::get_vendor_env;

    RestoreEnvironment legacy(kLegacyName);
    RestoreEnvironment primary(kPrimaryName);
    setenv(kLegacyName, "0", 1);
    setenv(kPrimaryName, "ON", 1);

    EXPECT_EQ(std::string(get_vendor_env(kPrimaryName)), "ON");
    EXPECT_TRUE(get_vendor_bool_env(kPrimaryName, true));
}

TEST(HygonVendorEnvTest, ScopedOverrideUsesAndRestoresPrimaryName) {
    using bytedance::hygon_vendor::detail::ScopedVendorEnvironment;
    using bytedance::hygon_vendor::detail::get_vendor_env;

    RestoreEnvironment legacy(kLegacyName);
    RestoreEnvironment primary(kPrimaryName);
    setenv(kLegacyName, "old", 1);
    unsetenv(kPrimaryName);

    {
        ScopedVendorEnvironment scoped;
        scoped.set(kPrimaryName, "route");

        EXPECT_EQ(std::string(get_vendor_env(kPrimaryName)), "route");
        EXPECT_STREQ(std::getenv(kLegacyName), "old");
    }

    EXPECT_EQ(std::getenv(kPrimaryName), nullptr);
    EXPECT_STREQ(std::getenv(kLegacyName), "old");
}

TEST(HygonVendorEnvTest, ScopedUnsetClearsBothNames) {
    using bytedance::hygon_vendor::detail::ScopedVendorEnvironment;

    RestoreEnvironment legacy(kLegacyName);
    RestoreEnvironment primary(kPrimaryName);
    setenv(kLegacyName, "old", 1);
    setenv(kPrimaryName, "primary-old", 1);

    {
        ScopedVendorEnvironment scoped;
        scoped.unset(kPrimaryName);

        EXPECT_EQ(std::getenv(kLegacyName), nullptr);
        EXPECT_EQ(std::getenv(kPrimaryName), nullptr);
    }

    ASSERT_NE(std::getenv(kLegacyName), nullptr);
    EXPECT_STREQ(std::getenv(kLegacyName), "old");
    ASSERT_NE(std::getenv(kPrimaryName), nullptr);
    EXPECT_STREQ(std::getenv(kPrimaryName), "primary-old");
}
