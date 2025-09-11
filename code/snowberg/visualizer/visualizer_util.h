#pragma once
#include "grassland/grassland.h"

namespace snowberg::visualizer {
using namespace grassland;

class Core;
class Scene;
class Mesh;
class Film;
class Camera;
class Program;
class Entity;

struct RenderContext;
class OwnershipHolder;

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 tex_coord;
  glm::vec4 color;
};

struct Material {
  glm::vec4 albedo;
};

struct CameraInfo {
  glm::mat4 proj;
  glm::mat4 view;
};

struct EntityInfo {
  Material material;
  glm::mat4 model;
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

typedef enum RenderStage { RENDER_STAGE_RASTER_GEOMETRY_PASS = 0, RENDER_STAGE_RASTER_LIGHTING_PASS = 1 } RenderStage;

typedef enum ProgramID : uint64_t {
  PROGRAM_ID_NO_NORMAL = 0,
  PROGRAM_AMBIENT_LIGHTING_PASS = 1,
  PROGRAM_DIRECTION_LIGHTING_PASS = 2,
} ProgramID;

}  // namespace snowberg::visualizer
