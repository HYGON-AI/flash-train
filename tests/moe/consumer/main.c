#include <stdbool.h>
#include <string.h>

#include <flash_train/flash_train.h>

int main(void) {
    FTrainAGGroupGemmConfig ag_config = FTRAIN_AG_GROUP_GEMM_CONFIG_INIT;
    bool available = false;
    if (ftrainMoeIsAvailable(&available) != FTRAIN_STATUS_SUCCESS || !available) { return 1; }
    if (strcmp(ag_config.route, "ag_nn_ring_chunk") != 0 || ag_config.topk != 1) { return 2; }
    FTrainMoeTensor empty = {0};
    if (ftrainAGGroupGemm(NULL, &ag_config, empty, empty, empty, empty, NULL) != FTRAIN_STATUS_INVALID_ARGUMENT) {
        return 4;
    }
    return 0;
}
