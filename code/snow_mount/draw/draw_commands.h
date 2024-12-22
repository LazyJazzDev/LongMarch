#pragma once
#include "snow_mount/draw/draw_util.h"

namespace snow_mount::draw {

class DrawCommand {
 public:
  virtual ~DrawCommand() = default;
  virtual void Execute(graphics::CommandContext *ctx) = 0;
};

class DrawCmdSetDrawRegion : public DrawCommand {
 public:
  DrawCmdSetDrawRegion(int x, int y, int width, int height);

  void Execute(graphics::CommandContext *ctx) override;

 private:
  int x_;
  int y_;
  int width_;
  int height_;
};

class DrawCmdDrawInstance : public DrawCommand {
 public:
  DrawCmdDrawInstance(Model *model,
                      graphics::Image *texture,
                      uint32_t instance_base,
                      uint32_t instance_count);

  void Execute(graphics::CommandContext *ctx) override;

 private:
  Model *model_;
  graphics::Image *texture_;
  uint32_t instance_base_;
  uint32_t instance_count_;
};

}  // namespace snow_mount::draw
