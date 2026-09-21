#include "sparkium/backend/device.h"

#include "sparkium/backend/graphics/graphics_device.h"
#include "stb_image.h"
#ifdef SPARKIUM_CPU_ENABLED
#include "sparkium/backend/cpu/cpu_device.h"
#ifdef SPARKIUM_CUDA_ENABLED
#include "sparkium/backend/cuda/cuda_device.h"
#endif
#endif

namespace sparkium {
using namespace grassland;

const char *BackendName(RenderBackend backend) {
  switch (backend) {
    case RenderBackend::Graphics:
      return "Graphics";
    case RenderBackend::CPU:
      return "CPU";
    case RenderBackend::CUDA:
      return "CUDA";
  }
  return "Unknown";
}

const char *BackendName(BackendSelection selection) {
  return selection.backend == RenderBackend::Graphics ? graphics::BackendAPIString(selection.graphics_api)
                                                      : BackendName(selection.backend);
}

graphics::BackendAPI ToGraphicsBackend(BackendSelection selection) {
  if (selection.backend != RenderBackend::Graphics)
    throw std::invalid_argument("render backend has no graphics API");
  return selection.graphics_api;
}

bool SupportBackend(BackendSelection selection) {
  switch (selection.backend) {
    case RenderBackend::CPU:
#ifdef SPARKIUM_CPU_ENABLED
      return true;
#else
      return false;
#endif
    case RenderBackend::CUDA:
#ifdef SPARKIUM_CUDA_ENABLED
      return true;
#else
      return false;
#endif
    case RenderBackend::Graphics:
      return graphics::SupportBackendAPI(selection.graphics_api);
  }
  return false;
}

int CreateDevice(BackendSelection selection,
                 const backend::Device::Settings &settings,
                 double_ptr<backend::Device> device) {
  if (!SupportBackend(selection))
    return -1;
  if (selection.backend == RenderBackend::CPU) {
#ifdef SPARKIUM_CPU_ENABLED
    device.construct<backend::CpuDevice>(settings);
    return 0;
#endif
  }
  if (selection.backend == RenderBackend::CUDA) {
#ifdef SPARKIUM_CUDA_ENABLED
    device.construct<backend::CudaDevice>(settings);
    return 0;
#endif
  }
  std::unique_ptr<graphics::Core> core;
  int result = graphics::CreateCore(selection.graphics_api, {settings.frames_in_flight, settings.enable_debug}, &core);
  if (result)
    return result;
  device.construct<backend::GraphicsDevice>(std::move(core));
  return 0;
}

int backend::Device::InitializeLogicalDeviceAutoSelect(bool require_ray_tracing) {
  int count = GetPhysicalDeviceProperties();
  if (count <= 0)
    return -1;
  std::vector<graphics::PhysicalDeviceProperties> properties(count);
  GetPhysicalDeviceProperties(properties.data());
  int selected = -1;
  for (int i = 0; i < count; ++i) {
    if (require_ray_tracing && !properties[i].ray_tracing_support)
      continue;
    if (selected < 0 || properties[i].score > properties[selected].score)
      selected = i;
  }
  return selected < 0 ? -1 : InitializeLogicalDevice(selected);
}

int backend::Device::LoadImage(const std::string &path, double_ptr<graphics::Image> image) {
  int width{}, height{}, channels{};
  // Preserve the existing graphics loader's byte-first decoding policy.
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> bytes(
      stbi_load(path.c_str(), &width, &height, &channels, 4), stbi_image_free);
  if (bytes) {
    int result = CreateImage(width, height, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, image);
    if (result)
      return result;
    image->UploadData(bytes.get());
    return 0;
  }
  std::unique_ptr<float, decltype(&stbi_image_free)> floats(stbi_loadf(path.c_str(), &width, &height, &channels, 4),
                                                            stbi_image_free);
  if (!floats)
    return -1;
  int result = CreateImage(width, height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, image);
  if (result)
    return result;
  image->UploadData(floats.get());
  return 0;
}
}  // namespace sparkium
