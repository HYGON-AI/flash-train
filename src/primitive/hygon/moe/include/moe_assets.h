#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include <hip/hip_runtime_api.h>

namespace ftrain::moe::assets {

struct MoeCodeObject final {
    std::string_view name;
    const unsigned char* data = nullptr;
    size_t size = 0;
};

MoeCodeObject getEmbeddedCodeObject(std::string_view name) noexcept;
std::string resolveCodeObject(std::string_view filename);
bool isCodeObjectAvailable(const std::string& path);
hipError_t loadCodeObject(hipModule_t* module, const std::string& path);

}
