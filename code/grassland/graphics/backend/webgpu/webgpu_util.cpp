#include "grassland/graphics/backend/webgpu/webgpu_util.h"

#include <stdexcept>

namespace grassland::graphics::backend {

wgpu::TextureFormat WebGPUFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_UNDEFINED:
      return wgpu::TextureFormat::Undefined;
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return wgpu::TextureFormat::BGRA8Unorm;
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return wgpu::TextureFormat::RGBA8Unorm;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return wgpu::TextureFormat::RGBA32Float;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return wgpu::TextureFormat::RG32Float;
    case IMAGE_FORMAT_R32_SFLOAT:
      return wgpu::TextureFormat::R32Float;
    case IMAGE_FORMAT_D32_SFLOAT:
      return wgpu::TextureFormat::Depth32Float;
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
      return wgpu::TextureFormat::RGBA16Float;
    case IMAGE_FORMAT_R32_UINT:
      return wgpu::TextureFormat::R32Uint;
    case IMAGE_FORMAT_R32_SINT:
      return wgpu::TextureFormat::R32Sint;
  }
  throw std::invalid_argument("unsupported WebGPU image format");
}

uint32_t WebGPUTexelSize(ImageFormat format) {
  return format == IMAGE_FORMAT_R32G32B32_SFLOAT ? 16 : PixelSize(format);
}

}  // namespace grassland::graphics::backend
