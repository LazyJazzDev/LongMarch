#include "sparkium/backend/cpu/texture.h"

#include "sparkium/backend/cpu/cpu_image.h"

namespace sparkium::backend::cpu {
Texture::Texture(Core *core, std::shared_ptr<const TextureData> data) : source(std::move(data)) {
  image = std::make_unique<backend::cpu::CpuImage>(source);
}
}  // namespace sparkium::backend::cpu
