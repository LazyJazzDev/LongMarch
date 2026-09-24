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
  DesktopGameSession(const std::string &name);
  ~DesktopGameSession();
  void Render();
  void ResetClock();
  void Resize(int width, int height);
  grassland::graphics::Core *Core() const;
  grassland::graphics::Image *Image() const;
  grassland::graphics::Window *Window() const;
  int FileRequest() const;
  std::string CompleteFile(const std::string &path);

 private:
  std::unique_ptr<life_demo::GameOfLife> life_;
  std::unique_ptr<puzzle_demo::TwentyFourEight> puzzle_;
};
