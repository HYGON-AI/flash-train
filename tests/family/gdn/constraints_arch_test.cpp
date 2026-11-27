// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <gtest/gtest.h>

#include "flash_train/constraints.hpp"
#include "flash_train/primitive/hygon/arch.hpp"

TEST(ConstraintsArch, RuntimeDeviceArch) {
    const ftrain::Constraints constraints{ftrain::getCurrentDeviceId()};
    EXPECT_FALSE(constraints.getArch().empty());
    EXPECT_EQ(constraints.getArch().find("gfx"), 0U);
    EXPECT_EQ(constraints.getArch().find(':'), std::string::npos);
}

TEST(ConstraintsArch, ArchitectureChangesKey) {
    const ftrain::Constraints gfx936{0, "gfx936"};
    const ftrain::Constraints gfx938{0, "gfx938"};
    EXPECT_NE(gfx936.getConstraintsKey(), gfx938.getConstraintsKey());
}

TEST(ConstraintsArch, Gfx936Gate) {
    EXPECT_TRUE(ftrain::isGfx936(ftrain::Constraints{0, "gfx936"}));
    EXPECT_FALSE(ftrain::isGfx936(ftrain::Constraints{0, "gfx938"}));
}

TEST(ConstraintsArch, Gfx938Gate) {
    EXPECT_TRUE(ftrain::isGfx938(ftrain::Constraints{0, "gfx938"}));
    EXPECT_FALSE(ftrain::isGfx938(ftrain::Constraints{0, "gfx936"}));
    EXPECT_FALSE(ftrain::isGfx938(ftrain::Constraints{0, "gfx940"}));
}
