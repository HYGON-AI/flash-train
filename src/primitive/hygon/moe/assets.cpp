#include "moe_assets.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace ftrain::moe::assets {
namespace {

constexpr std::string_view embedded_prefix = "embedded://gfx936/";

bool isEmbedded(std::string_view path) {
    return path.substr(0, embedded_prefix.size()) == embedded_prefix;
}

}

std::string resolveCodeObject(std::string_view filename) {
    const char* override_directory = std::getenv("FTRAIN_MOE_ASM_DIR");
    if (override_directory != nullptr && *override_directory != '\0') {
        const std::filesystem::path directory(override_directory);
        if (!std::filesystem::is_directory(directory)) {
            throw std::runtime_error("FTRAIN_MOE_ASM_DIR does not name a directory");
        }
        return (directory / filename).string();
    }
    return std::string(embedded_prefix) + std::string(filename);
}

bool isCodeObjectAvailable(const std::string& path) {
    if (isEmbedded(path)) {
        return getEmbeddedCodeObject(std::string_view(path).substr(embedded_prefix.size())).data != nullptr;
    }
    return std::filesystem::is_regular_file(path);
}

hipError_t loadCodeObject(hipModule_t* module, const std::string& path) {
    if (module == nullptr) { return hipErrorInvalidValue; }
    if (isEmbedded(path)) {
        const auto object = getEmbeddedCodeObject(std::string_view(path).substr(embedded_prefix.size()));
        if (object.data == nullptr) { return hipErrorFileNotFound; }
        return hipModuleLoadData(module, object.data);
    }
    return hipModuleLoad(module, path.c_str());
}

}
