#ifndef FTRAIN_OPERATION_OPERATION_HPP_
#define FTRAIN_OPERATION_OPERATION_HPP_

#include <variant>

#include "flash_train/operation/gemm.hpp"

namespace ftrain {

using OperationAttributes =
    std::variant<GemmAttributes, GroupedABCDGemmAttributes, GroupedBCDGemmAttributes, GroupedABGemmAttributes>;

}  // namespace ftrain

#endif
