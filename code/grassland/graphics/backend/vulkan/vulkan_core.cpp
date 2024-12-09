#include "grassland/graphics/backend/vulkan/vulkan_core.h"

namespace grassland::graphics::backend {

VulkanCore::VulkanCore(const Settings &settings) : Core(settings) {
}

int VulkanCore::CreateBuffer(size_t size,
                             BufferType type,
                             double_ptr<Buffer> pp_buffer) {
  return 0;
}

int VulkanCore::CreateImage(int width,
                            int height,
                            ImageFormat format,
                            double_ptr<Image> pp_image) {
  return 0;
}

}  // namespace grassland::graphics::backend
