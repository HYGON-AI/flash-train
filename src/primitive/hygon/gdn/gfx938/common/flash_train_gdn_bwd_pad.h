#pragma once

#include "gfx938_device_symbols.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Zero-padding / cropping copy used by the backward entry for shapes the stage kernels do not cover
// natively (S % 64 != 0, K or V not in {64,128}).  Each descriptor is a dense 4-D tensor copy
//   dst[i0][i1][i2][i3] = (i0 < a0 && i1 < a1 && i2 < a2 && i3 < a3) ? src[i0][i1][i2][i3] : 0
// for every (i0..i3) inside the dst extents d0..d3; src is dense with extents a0..a3, dst is dense
// with extents d0..d3.  Padding is d >= a (the missing part is written as zero), cropping is d <= a.
// With swap23 = 1 the last two dimensions of the source are read transposed:
//   dst[i0][i1][i2][i3] = (i0 < a0 && i1 < a1 && i3 < a2 && i2 < a3) ? src[i0][i1][i3][i2] : 0
// (src still dense with extents a0..a3 in its own order; dst extents d2/d3 correspond to a3/a2).
// elem is the element size in bytes (2 or 4).  Up to FLASH_TRAIN_GDN_BWD_PAD_MAX_DESCS descriptors go
// out in one launch (grid.y = descriptor index).  Returns 0 ok, 1 null pointer, 2 bad argument, 5 HIP error.
#define FLASH_TRAIN_GDN_BWD_PAD_MAX_DESCS 8

typedef struct FlashTrainGDNBwdPadDesc {
    const void* src;
    void* dst;
    int elem;
    int swap23;  // 1: transpose the last two dims while copying
    int a[4];    // src extents
    int d[4];    // dst extents
} FlashTrainGDNBwdPadDesc;

int flash_train_gdn_bwd_pad_copy(const FlashTrainGDNBwdPadDesc* descs, int n, void* stream);

#ifdef __cplusplus
}
#endif
