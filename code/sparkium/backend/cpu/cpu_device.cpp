#include "sparkium/backend/cpu/cpu_device.h"

namespace sparkium::backend {
int CpuDevice::GetPhysicalDeviceProperties(PhysicalDeviceProperties *p) {
  if (p) {
    p[0].name = "Native CPU (Slang LLVM JIT)";
    p[0].score = 1;
    p[0].ray_tracing_support = false;
    p[0].geometry_shader_support = false;
  }
  return 1;
}

int CpuDevice::InitializeLogicalDevice(int index) {
  if (index != 0)
    return -1;
  PhysicalDeviceProperties properties;
  GetPhysicalDeviceProperties(&properties);
  device_name_ = properties.name;
  return 0;
}
}  // namespace sparkium::backend
