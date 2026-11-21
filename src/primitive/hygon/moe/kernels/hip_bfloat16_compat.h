#pragma once

#if __has_include(<hip/hip_bf16.h>)
#include <hip/hip_bf16.h>
#else
#include <hip/hip_bfloat16.h>
#endif
