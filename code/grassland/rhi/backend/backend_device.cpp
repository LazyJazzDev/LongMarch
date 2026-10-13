#include "grassland/rhi/backend/backend_device.h"

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi::backend {

IDevice::IDevice(PhysicalDevice info, const ObjectPtr<ICore> &core) : core_(core), info_(std::move(info)) {
}

IDevice::~IDevice() = default;

}  // namespace grassland::rhi::backend
