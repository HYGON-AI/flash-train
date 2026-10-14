#ifndef FTRAIN_HYGON_COMM_INFO_H
#define FTRAIN_HYGON_COMM_INFO_H
namespace bytedance
{
namespace hygon_vendor
{

enum class HygonQuantMode
{
    UnQuant = 0,
    PerTensor = 1,
    PerChannel = 2,
    PerBlock = 3,
};

} // namespace hygon_vendor
} // namespace bytedance

#endif // FTRAIN_HYGON_COMM_INFO_H
