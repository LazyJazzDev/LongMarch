#include "optix_device.h"

#include <optix_function_table_definition.h>

#include <iostream>
#include <mutex>

#include "optix_util.h"
#if OPTIX_VERSION < 80000
#error "LongMarch requires OptiX SDK 8.0 or newer"
#endif

namespace grassland::graphics::backend {

namespace {
void LogCallback(unsigned int level, const char *tag, const char *message, void *) {
  std::cerr << "OptiX [" << level << "][" << tag << "] " << message << '\n';
}
}  // namespace

OptixDevice::OptixDevice(CUcontext context, bool debug) {
  static std::once_flag initialized;
  std::call_once(initialized, [] { CheckOptix(optixInit(), "initialize driver (check NVIDIA driver installation)"); });
  OptixDeviceContextOptions options{};
  options.logCallbackFunction = LogCallback;
  options.logCallbackLevel = debug ? 4 : 2;
  options.validationMode = debug ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
  CheckOptix(optixDeviceContextCreate(context, &options, &context_), "create device context");
  try {
    uint32_t rtcore_version{};
    CheckOptix(optixDeviceContextGetProperty(context_, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION, &rtcore_version,
                                             sizeof(rtcore_version)),
               "query hardware traversal");
    if (!rtcore_version)
      throw std::runtime_error("OptiX hardware traversal requires an NVIDIA GPU with RT cores");
  } catch (...) {
    optixDeviceContextDestroy(context_);
    context_ = nullptr;
    throw;
  }
}

OptixDevice::~OptixDevice() {
  if (context_)
    optixDeviceContextDestroy(context_);
}

}  // namespace grassland::graphics::backend
