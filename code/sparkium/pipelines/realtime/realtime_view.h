#pragma once
#include "sparkium/pipelines/common/core/core_util.h"

namespace sparkium::render_shared {
class SoftwarePipeline;
}

namespace sparkium::realtime {

// Per-view history, separate from the scene's shared material and tracing data.
class RealtimeView : public Object {
 public:
  explicit RealtimeView(sparkium::Film &film);
  void Begin(graphics::CommandContext *commands,
             render_shared::SoftwarePipeline *pipeline,
             const std::vector<graphics::Buffer *> &buffers,
             render_shared::Camera *camera,
             uint32_t width,
             uint32_t height,
             int scale,
             int history,
             int updates,
             uint32_t frame);
  void BindTrace(graphics::CommandContext *commands);
  void Resolve(graphics::CommandContext *commands,
               graphics::Image *output,
               render_shared::SoftwarePipeline *pipeline,
               const std::vector<graphics::Buffer *> &buffers);

  graphics::Buffer *ParametersBuffer() const {
    return parameters_buffer_.get();
  }

  uint32_t Width() const {
    return low_width_;
  }

  uint32_t Height() const {
    return low_height_;
  }

 private:
  struct Parameters {
    glm::mat4 view_projection{1.0f};
    glm::mat4 previous_view_projection{1.0f};
    glm::uvec4 extent{};
    glm::uvec4 config{};
    glm::vec4 camera_position{};
  } parameters_;

  render_shared::Core *core_;
  uint32_t width_{}, height_{}, low_width_{}, low_height_{}, buffer_count_{};
  int index_{};
  uint32_t frame_index_{};
  uint32_t vertex_count_{};
  std::vector<uint32_t> vertex_counts_;
  std::unique_ptr<graphics::Buffer> draw_ranges_;
  bool valid_{};
  std::unique_ptr<graphics::Image> visibility_, depth_, filtered_;
  std::unique_ptr<graphics::Image> color_[2], geometry_[2], normal_[2];
  std::unique_ptr<graphics::Buffer> parameters_buffer_;
  std::unique_ptr<graphics::Shader> vertex_, pixel_, resolve_shader_, filter_shader_;
  std::unique_ptr<graphics::Program> visibility_program_;
  std::unique_ptr<graphics::ComputeProgram> resolve_program_, filter_program_;
};
}  // namespace sparkium::realtime
