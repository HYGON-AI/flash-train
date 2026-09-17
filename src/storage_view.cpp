#include "flash_train/error.hpp"
#include "flash_train/storage_view.hpp"

namespace ftrain {

StorageView::StorageView(const FTrainStorageView& storage_view)
    : memory_(storage_view.memory), numeric_type_(storage_view.numeric_type), index_type_(storage_view.index_type),
      is_host_memory_(storage_view.is_host_memory) {
    if (numeric_type_ == FTRAIN_NUMERIC_TYPE_INVALID || numeric_type_ >= FTRAIN_NUMERIC_TYPE_COUNT) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "StorageView numeric_type %u is invalid",
                        static_cast<unsigned int>(numeric_type_));
    }

    if (index_type_ >= FTRAIN_INDEX_TYPE_COUNT) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "StorageView index_type %u is unknown",
                        static_cast<unsigned int>(index_type_));
    }

    const std::size_t num_dims = storage_view.num_dims;
    if (num_dims == 0 && storage_view.memory == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "StorageView memory must not be null for a scalar");
    }

    if (num_dims != 0 && storage_view.dims == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "StorageView dims must not be null when num_dims is %zu",
                        num_dims);
    }

    if (storage_view.strides == nullptr && index_type_ == FTRAIN_INDEX_TYPE_INVALID) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "StorageView requires strides or a non-invalid index_type");
    }
    if (storage_view.strides != nullptr && index_type_ != FTRAIN_INDEX_TYPE_INVALID) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "StorageView strides must be null when index_type %u defines a predefined layout",
                        static_cast<unsigned int>(index_type_));
    }

    for (std::size_t index = 0; index < num_dims; ++index) {
        if (storage_view.dims[index] < 0) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "StorageView dimension %zu must be non-negative, got %lld",
                            index, static_cast<long long>(storage_view.dims[index]));
        }
    }

    if (num_dims == 0) { return; }

    dims_.assign(storage_view.dims, storage_view.dims + num_dims);
    if (storage_view.strides != nullptr) { strides_.assign(storage_view.strides, storage_view.strides + num_dims); }
}

}  // namespace ftrain
