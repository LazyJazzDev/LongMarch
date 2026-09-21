#include "sparkium/backend/cuda/texture.h"

namespace sparkium::backend::cuda {
Texture::Texture(Core *core, std::shared_ptr<const TextureData> data) : source(std::move(data)) {
  if (core->BackendDevice()->CreateImage(source->width, source->height, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image))
    throw std::runtime_error("failed to allocate backend texture");
  image->UploadData(source->rgba.data());
}
}  // namespace sparkium::backend::cuda
