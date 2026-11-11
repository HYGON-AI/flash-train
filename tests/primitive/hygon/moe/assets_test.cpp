#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "moe_assets.h"

namespace {

namespace assets = ftrain::moe::assets;

class MoeAssetsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const char* value = std::getenv("FTRAIN_MOE_ASM_DIR");
        if (value != nullptr) { previous_directory = value; }
        unsetenv("FTRAIN_MOE_ASM_DIR");
        std::string pattern = (std::filesystem::temp_directory_path() / "ftrain-moe-assets-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const char* directory = mkdtemp(buffer.data());
        ASSERT_NE(directory, nullptr);
        temporary_directory = directory;
    }

    void TearDown() override {
        if (previous_directory.has_value()) { setenv("FTRAIN_MOE_ASM_DIR", previous_directory->c_str(), 1); }
        else { unsetenv("FTRAIN_MOE_ASM_DIR"); }
        if (!temporary_directory.empty()) { std::filesystem::remove_all(temporary_directory); }
    }

    bool hasGfx936() const {
        int device = -1;
        hipDeviceProp_t properties{};
        return hipGetDevice(&device) == hipSuccess && hipGetDeviceProperties(&properties, device) == hipSuccess &&
               std::string_view(properties.gcnArchName).substr(0, 6) == "gfx936";
    }

    std::optional<std::string> previous_directory;
    std::filesystem::path temporary_directory;
};

TEST_F(MoeAssetsTest, EmbeddedObjectsExactlyMatchAssemblerOutputs) {
    size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(FTRAIN_MOE_TEST_CODE_OBJECT_DIR)) {
        if (entry.path().extension() != ".co") { continue; }
        const auto name = entry.path().filename().string();
        SCOPED_TRACE(name);
        const auto object = assets::getEmbeddedCodeObject(name);
        ASSERT_NE(object.data, nullptr);
        std::ifstream input(entry.path(), std::ios::binary);
        const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input), {}};
        ASSERT_EQ(object.size, bytes.size());
        EXPECT_EQ(object.name, name);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(object.data) % 4096, 0);
        EXPECT_EQ(std::memcmp(object.data, bytes.data(), bytes.size()), 0);
        ++count;
    }
    EXPECT_EQ(count, 10);
}

TEST_F(MoeAssetsTest, DefaultResolutionNeedsNoExternalFile) {
    const auto path = assets::resolveCodeObject("ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co");
    EXPECT_EQ(path, "embedded://gfx936/ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co");
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_TRUE(assets::isCodeObjectAvailable(path));
    EXPECT_EQ(std::filesystem::path(path).stem(), "ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal");
    setenv("FTRAIN_MOE_ASM_DIR", "", 1);
    EXPECT_EQ(assets::resolveCodeObject("ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co"), path);
}

TEST_F(MoeAssetsTest, ExplicitDirectoryOverrideUsesExternalFiles) {
    setenv("FTRAIN_MOE_ASM_DIR", temporary_directory.c_str(), 1);
    const auto path = assets::resolveCodeObject("ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co");
    EXPECT_EQ(path, (temporary_directory / "ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co").string());
    EXPECT_FALSE(assets::isCodeObjectAvailable(path));
    {
        std::ofstream output(path, std::ios::binary);
        output << "test code object";
    }
    EXPECT_TRUE(assets::isCodeObjectAvailable(path));
    const auto directory = temporary_directory / "directory.co";
    std::filesystem::create_directory(directory);
    EXPECT_FALSE(assets::isCodeObjectAvailable(directory.string()));
}

TEST_F(MoeAssetsTest, InvalidExplicitDirectoryDoesNotSilentlyFallBack) {
    setenv("FTRAIN_MOE_ASM_DIR", (temporary_directory / "missing").c_str(), 1);
    EXPECT_THROW(assets::resolveCodeObject("ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co"), std::runtime_error);
}

TEST_F(MoeAssetsTest, MissingEmbeddedObjectsAndNullModulesAreRejected) {
    const auto object = assets::getEmbeddedCodeObject("missing.co");
    EXPECT_EQ(object.data, nullptr);
    EXPECT_EQ(object.size, 0);
    EXPECT_FALSE(assets::isCodeObjectAvailable(assets::resolveCodeObject("missing.co")));
    EXPECT_FALSE(assets::isCodeObjectAvailable("embedded://gfx936/../ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co"));
    hipModule_t module = nullptr;
    EXPECT_EQ(assets::loadCodeObject(&module, assets::resolveCodeObject("missing.co")), hipErrorFileNotFound);
    EXPECT_EQ(module, nullptr);
    EXPECT_EQ(assets::loadCodeObject(nullptr, assets::resolveCodeObject("ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co")), hipErrorInvalidValue);
}

TEST_F(MoeAssetsTest, LoadsEveryEmbeddedObjectAndFindsItsKernel) {
    if (!hasGfx936()) { GTEST_SKIP() << "Embedded MoE kernels require gfx936"; }
    size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(FTRAIN_MOE_TEST_CODE_OBJECT_DIR)) {
        if (entry.path().extension() != ".co") { continue; }
        SCOPED_TRACE(entry.path().filename().string());
        hipModule_t module = nullptr;
        ASSERT_EQ(assets::loadCodeObject(&module, assets::resolveCodeObject(entry.path().filename().string())),
                  hipSuccess);
        hipFunction_t function = nullptr;
        EXPECT_EQ(hipModuleGetFunction(&function, module, entry.path().stem().c_str()), hipSuccess);
        EXPECT_NE(function, nullptr);
        EXPECT_EQ(hipModuleUnload(module), hipSuccess);
        ++count;
    }
    EXPECT_EQ(count, 10);
}

TEST_F(MoeAssetsTest, ExplicitOverrideLoadsFromFileAndMissingFilesDoNotFallBack) {
    if (!hasGfx936()) { GTEST_SKIP() << "Embedded MoE kernels require gfx936"; }
    setenv("FTRAIN_MOE_ASM_DIR", temporary_directory.c_str(), 1);
    const auto path = assets::resolveCodeObject("ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co");
    hipModule_t module = nullptr;
    EXPECT_EQ(assets::loadCodeObject(&module, path), hipErrorFileNotFound);
    EXPECT_EQ(module, nullptr);
    std::filesystem::copy_file(std::filesystem::path(FTRAIN_MOE_TEST_CODE_OBJECT_DIR) / "ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal.co", path);
    ASSERT_EQ(assets::loadCodeObject(&module, path), hipSuccess);
    hipFunction_t function = nullptr;
    EXPECT_EQ(hipModuleGetFunction(&function, module, "ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nn_a2a_direct_output_nchunk_signal"), hipSuccess);
    EXPECT_NE(function, nullptr);
    EXPECT_EQ(hipModuleUnload(module), hipSuccess);
}

}
