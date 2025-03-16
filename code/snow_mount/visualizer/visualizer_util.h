#pragma once
#include "grassland/grassland.h"

namespace snow_mount::visualizer {
using namespace grassland;

class Core;
class Scene;
class Mesh;
class Film;
struct Camera;
class Program;

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 tex_coord;
  glm::vec4 color;
};

struct EntityInfo {
  glm::mat4 model;
};

struct Material {
  glm::vec4 albedo;
};

typedef enum TextureType { TEXTURE_TYPE_SDR = 0, TEXTURE_TYPE_HDR = 1 } TextureType;

graphics::ImageFormat TextureTypeToImageFormat(TextureType type);

typedef enum FilmChannel {
  FILM_CHANNEL_EXPOSURE = 0,
  FILM_CHANNEL_ALBEDO = 1,
  FILM_CHANNEL_POSITION = 2,
  FILM_CHANNEL_NORMAL = 3,
  FILM_CHANNEL_DEPTH = 4,
  FILM_CHANNEL_COUNT = 5
} FilmChannel;

graphics::ImageFormat FilmChannelImageFormat(FilmChannel channel);

}  // namespace snow_mount::visualizer
