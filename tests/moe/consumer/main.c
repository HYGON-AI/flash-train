#include <stdbool.h>
#include <string.h>

#include <flash_train/flash_train.h>

int main(void) {
    FTrainAGGroupGemmConfig ag_config;
    FTrainMoEReduceRSConfig rs_config;
    bool available = false;
    if (ftrainMoeIsAvailable(&available) != FTRAIN_STATUS_SUCCESS || !available) { return 1; }
    if (ftrainAGGroupGemmConfigInit(&ag_config) != FTRAIN_STATUS_SUCCESS ||
        strcmp(ag_config.route, "ag_nn_ring_chunk") != 0) { return 2; }
    if (ftrainMoEReduceRSConfigInit(&rs_config) != FTRAIN_STATUS_SUCCESS ||
        strcmp(rs_config.route, "rs_nn") != 0) { return 3; }
    return 0;
}
