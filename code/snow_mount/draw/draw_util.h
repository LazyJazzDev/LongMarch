#pragma once
#include "grassland/grassland.h"

namespace snow_mount::draw {

using namespace grassland;

class Core;
class Texture;
class Model;

struct Vertex {
  glm::vec2 position;
  glm::vec2 tex_coord;
  glm::vec2 color;
};

}  // namespace snow_mount::draw
