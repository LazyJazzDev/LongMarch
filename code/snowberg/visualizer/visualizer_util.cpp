#include "snowberg/visualizer/visualizer_util.h"

namespace snowberg::visualizer {

graphics::ImageFormat TextureTypeToImageFormat(TextureType type) {
  switch (type) {
    case TEXTURE_TYPE_HDR:
      return graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT;
    case TEXTURE_TYPE_SDR:
      return graphics::IMAGE_FORMAT_R8G8B8A8_UNORM;
  }
  return graphics::IMAGE_FORMAT_R8G8B8A8_UNORM;
}

graphics::ImageFormat FilmChannelImageFormat(FilmChannel channel) {
  switch (channel) {
    case FILM_CHANNEL_EXPOSURE:
    case FILM_CHANNEL_ALBEDO:
    case FILM_CHANNEL_POSITION:
    case FILM_CHANNEL_NORMAL:
      return graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT;
    case FILM_CHANNEL_DEPTH:
      return graphics::IMAGE_FORMAT_D32_SFLOAT;
    default:
      throw std::runtime_error("Invalid FilmChannel");
  }
}

}  // namespace snowberg::visualizer
