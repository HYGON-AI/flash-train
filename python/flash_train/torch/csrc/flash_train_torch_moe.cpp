#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <flash_train/flash_train.h>

namespace py = pybind11;

namespace {

class DeviceGuard final {
  public:
    explicit DeviceGuard(int device) {
        if (hipGetDevice(&previous_) != hipSuccess || hipSetDevice(device) != hipSuccess) {
            throw std::runtime_error("Cannot select the MoE HIP device");
        }
    }
    ~DeviceGuard() { static_cast<void>(hipSetDevice(previous_)); }

  private:
    int previous_ = 0;
};

using TensorDescription = std::tuple<uintptr_t, std::vector<int64_t>, int, uintptr_t, uint64_t>;

FTrainMoeTensor makeTensor(const TensorDescription& description) {
    const auto& [address, dimensions, type, identity, version] = description;
    const auto dtype                                           = type == 0   ? FTRAIN_NUMERIC_TYPE_BF16
                                                                 : type == 1 ? FTRAIN_NUMERIC_TYPE_INT32
                                                                             : FTRAIN_NUMERIC_TYPE_INT64;
    if (type < 0 || type > 2 || dimensions.size() > 255) {
        throw std::invalid_argument("Invalid MoE tensor description");
    }
    return {
        {reinterpret_cast<void*>(address), dimensions.data(), nullptr, static_cast<uint8_t>(dimensions.size()), dtype,
         FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        reinterpret_cast<void*>(identity),
        version
    };
}

class MoeSession final {
  public:
    MoeSession(bool is_ag, int rank, int world_size, int device, py::object gather, int m_dim, int n_dim, int k_dim,
               int experts, int topk, int capacity_m, const std::string& route, const std::vector<int64_t>& sizes,
               int n_chunks, uintptr_t stream)
        : is_ag_(is_ag), device_(device), world_size_(world_size), gather_(std::move(gather)) {
        DeviceGuard guard(device_);
        FTrainCollectiveContext context = nullptr;
        check(ftrainCollectiveContextCreate(&context, rank, world_size, device_, allGather, this));
        FTrainStatus status;
        {
            py::gil_scoped_release release;
            if (is_ag_) {
                FTrainAGGroupGemmConfig config{};
                ftrainAGGroupGemmConfigInit(&config);
                config.m_dim            = m_dim;
                config.n_dim            = n_dim;
                config.k_dim            = k_dim;
                config.num_experts      = experts;
                config.topk             = topk;
                config.capacity_m       = capacity_m;
                config.route            = route.c_str();
                config.rank_m_sizes     = sizes.empty() ? nullptr : sizes.data();
                config.num_rank_m_sizes = sizes.size();
                status = ftrainAGGroupGemmCreate(&ag_, context, &config, reinterpret_cast<FTrainStream>(stream));
            } else {
                FTrainMoEReduceRSConfig config{};
                ftrainMoEReduceRSConfigInit(&config);
                config.m_dim            = m_dim;
                config.n_dim            = n_dim;
                config.k_dim            = k_dim;
                config.num_experts      = experts;
                config.topk             = topk;
                config.capacity_m       = capacity_m;
                config.route            = route.c_str();
                config.rank_m_sizes     = sizes.empty() ? nullptr : sizes.data();
                config.num_rank_m_sizes = sizes.size();
                config.n_chunks         = n_chunks;
                status = ftrainMoEReduceRSCreate(&rs_, context, &config, reinterpret_cast<FTrainStream>(stream));
            }
        }
        const std::string message = ftrainGetLastMessage();
        static_cast<void>(ftrainCollectiveContextDestroy(context));
        check(status, message);
    }

    ~MoeSession() {
        try {
            close();
        }
        catch (...) {
        }
    }

    void execute(const std::vector<TensorDescription>& tensors, uintptr_t stream) {
        requireOpen();
        if (tensors.size() != (is_ag_ ? 4U : 5U)) { throw std::invalid_argument("Invalid MoE tensor count"); }
        DeviceGuard guard(device_);
        callback_error_.clear();
        FTrainStatus status;
        {
            py::gil_scoped_release release;
            if (is_ag_) {
                status = ftrainAGGroupGemmExecute(ag_, makeTensor(tensors[0]), makeTensor(tensors[1]),
                                                  makeTensor(tensors[2]), makeTensor(tensors[3]),
                                                  reinterpret_cast<FTrainStream>(stream));
            } else {
                status = ftrainMoEReduceRSExecute(rs_, makeTensor(tensors[0]), makeTensor(tensors[1]),
                                                  makeTensor(tensors[2]), makeTensor(tensors[3]),
                                                  makeTensor(tensors[4]), reinterpret_cast<FTrainStream>(stream));
            }
        }
        check(status);
    }

    void setRankMSizes(int m_dim, const std::vector<int64_t>& sizes) {
        requireOpen();
        DeviceGuard guard(device_);
        callback_error_.clear();
        FTrainStatus status;
        {
            py::gil_scoped_release release;
            const auto* data = sizes.empty() ? nullptr : sizes.data();
            status           = is_ag_ ? ftrainAGGroupGemmSetRankMSizes(ag_, m_dim, data, sizes.size())
                                      : ftrainMoEReduceRSSetRankMSizes(rs_, m_dim, data, sizes.size());
        }
        check(status);
    }

    void synchronize() {
        requireOpen();
        DeviceGuard guard(device_);
        FTrainStatus status;
        {
            py::gil_scoped_release release;
            status = is_ag_ ? ftrainAGGroupGemmSynchronize(ag_) : ftrainMoEReduceRSSynchronize(rs_);
        }
        check(status);
    }

    void close() {
        if (ag_ == nullptr && rs_ == nullptr) { return; }
        DeviceGuard guard(device_);
        const auto status = is_ag_ ? ftrainAGGroupGemmDestroy(ag_) : ftrainMoEReduceRSDestroy(rs_);
        check(status);
        ag_ = nullptr;
        rs_ = nullptr;
    }

  private:
    void requireOpen() const {
        if (ag_ == nullptr && rs_ == nullptr) { throw std::runtime_error("MoE operator is closed"); }
    }

    void check(FTrainStatus status, const std::string& message = {}) const {
        if (status != FTRAIN_STATUS_SUCCESS) {
            throw std::runtime_error("MoE C API failed (status " + std::to_string(status) +
                                     "): " + (message.empty() ? ftrainGetLastMessage() : message) +
                                     (callback_error_.empty() ? "" : "; bootstrap: " + callback_error_));
        }
    }

    static FTrainStatus allGather(void* state, const void* send, void* receive, uint64_t bytes) {
        auto& session = *static_cast<MoeSession*>(state);
        py::gil_scoped_acquire acquire;
        try {
            const auto result = session.gather_(py::bytes(static_cast<const char*>(send), bytes));
            if (!py::isinstance<py::bytes>(result)) { throw std::runtime_error("allgather must return bytes"); }
            const auto values = py::cast<std::string>(result);
            if (values.size() != bytes * session.world_size_) {
                throw std::runtime_error("allgather returned an invalid byte count");
            }
            std::memcpy(receive, values.data(), values.size());
            return FTRAIN_STATUS_SUCCESS;
        }
        catch (const std::exception& error) {
            session.callback_error_ = error.what();
            return FTRAIN_STATUS_INTERNAL_ERROR;
        }
    }

    bool is_ag_;
    int device_;
    int world_size_;
    py::object gather_;
    std::string callback_error_;
    FTrainAGGroupGemm ag_ = nullptr;
    FTrainMoEReduceRS rs_ = nullptr;
};

}

void bindMoe(py::module_& module) {
    module.def("moe_is_available", [] {
        bool available = false;
        if (ftrainMoeIsAvailable(&available) != FTRAIN_STATUS_SUCCESS) {
            throw std::runtime_error(ftrainGetLastMessage());
        }
        return available;
    });
    py::class_<MoeSession>(module, "_MoeSession")
        .def(py::init<bool, int, int, int, py::object, int, int, int, int, int, int, const std::string&,
                      const std::vector<int64_t>&, int, uintptr_t>())
        .def("execute", &MoeSession::execute)
        .def("set_rank_m_sizes", &MoeSession::setRankMSizes)
        .def("synchronize", &MoeSession::synchronize)
        .def("close", &MoeSession::close);
}
