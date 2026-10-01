#pragma once
#include "grassland/graphics/graphics.h"

namespace life_demo {
class GameOfLife;
}

namespace puzzle_demo {
class TwentyFourEight;
}

class DesktopGameSession {
 public:
  DesktopGameSession(const std::string &name,
                     grassland::graphics::BackendAPI backend = grassland::graphics::BACKEND_API_DEFAULT);
  ~DesktopGameSession();
  void Render();
  double NextFrameDelay() const;
  void ResetClock();
  void PrepareInput(bool sleeping);
  void Resize(int width, int height);
  void SetIconOrientation(float radians);
  void SetBottomControlInset(float height_fraction);
  void SetCutoutInsets(float left, float top, float right);
  void SetControlExtentLimit(float height_fraction);
  void EnableNativeSizeControls();
  glm::ivec2 TakeSizeControlRequest();
  glm::vec4 SizeControlBounds(int axis) const;
  void SetGridDimension(int axis, int value);
  grassland::graphics::Core *Core() const;
  grassland::graphics::Image *Image() const;
  grassland::graphics::Window *Window() const;
  int FileRequest() const;
  std::string CompleteFile(const std::string &path);

 private:
  std::unique_ptr<life_demo::GameOfLife> life_;
  std::unique_ptr<puzzle_demo::TwentyFourEight> puzzle_;
};
