#ifndef FTRAIN_PRIMITIVE_PRIMITIVE_HPP_
#define FTRAIN_PRIMITIVE_PRIMITIVE_HPP_

// Convenience header for the primitive layer: the Primitive interface
// plus every concrete primitive implementation. A family registers its
// records through this one header, and a new implementation joins the
// library by adding its include here plus its entry in the family's
// makeRecords().

#include "flash_train/primitive/base.hpp"
#include "flash_train/primitive/gemm/fp32_gemm.hpp"

#endif
