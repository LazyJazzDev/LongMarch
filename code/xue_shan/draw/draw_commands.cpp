#include "xue_shan/draw/draw_commands.h"

#include "xue_shan/draw/draw_model.h"
#include "xue_shan/draw/draw_texture.h"

namespace XS::draw {

DrawCmdSetDrawRegion::DrawCmdSetDrawRegion(int x, int y, int width, int height)
    : x_(x), y_(y), width_(width), height_(height) {
}

void DrawCmdSetDrawRegion::Execute(graphics::CommandContext *ctx) {
  graphics::Viewport viewport;
  graphics::Scissor scissor;
  viewport.x = x_;
  viewport.y = y_;
  viewport.width = width_;
  viewport.height = height_;
  viewport.min_depth = 0.0f;
  viewport.max_depth = 1.0f;
  scissor.offset = {x_, y_};
  scissor.extent.width = width_;
  scissor.extent.height = height_;
  ctx->CmdSetViewport(viewport);
  ctx->CmdSetScissor(scissor);
}

DrawCmdDrawInstance::DrawCmdDrawInstance(Model *model,
                                         graphics::Image *texture,
                                         uint32_t instance_base,
                                         uint32_t instance_count)
    : model_(model), texture_(texture), instance_base_(instance_base), instance_count_(instance_count) {
}

void DrawCmdDrawInstance::Execute(graphics::CommandContext *ctx) {
  ctx->CmdBindResources(1, {texture_});
  ctx->CmdBindVertexBuffers(0, {model_->VertexBuffer()}, {0});
  ctx->CmdBindIndexBuffer(model_->IndexBuffer(), 0);
  ctx->CmdDrawIndexed(model_->IndexCount(), instance_count_, 0, 0, instance_base_);
}

}  // namespace XS::draw
