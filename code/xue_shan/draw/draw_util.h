#pragma once
#include "cao_di/grassland.h"

namespace XS::draw {

using namespace CD;

class Core;
class Texture;
class Model;
class DrawCommand;
class FontCore;

struct Vertex {
  glm::vec2 position;
  glm::vec2 tex_coord;
  glm::vec4 color;
};

typedef glm::mat4 Transform;

struct DrawMetadata {
  Transform transform;
  glm::vec4 color;
};

Transform PixelCoordToNDC(int width, int height);

}  // namespace XS::draw
