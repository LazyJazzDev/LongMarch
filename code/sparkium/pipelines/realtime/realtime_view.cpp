#include "sparkium/pipelines/realtime/realtime_view.h"

#include <algorithm>
#include <glm/ext/matrix_clip_space.hpp>
#include <limits>

#include "grassland/graphics/frame_profile.h"
#include "sparkium/core/camera.h"
#include "sparkium/pipelines/common/core/camera.h"
#include "sparkium/pipelines/common/core/core.h"
#include "sparkium/pipelines/common/core/software_pipeline.h"

namespace sparkium::realtime {
RealtimeView::RealtimeView(sparkium::Film &film) : core_(render_shared::DedicatedCast(film.GetCore())) {
  core_->GraphicsCore()->CreateBuffer(sizeof(Parameters), graphics::BUFFER_TYPE_STATIC, &parameters_buffer_);
}

void RealtimeView::Begin(graphics::CommandContext *commands,
                         render_shared::SoftwarePipeline *pipeline,
                         const std::vector<graphics::Buffer *> &buffers,
                         render_shared::Camera *camera,
                         uint32_t width,
                         uint32_t height,
                         int scale,
                         int history,
                         int updates,
                         uint32_t frame) {
  auto *graphics = core_->GraphicsCore();
  scale = std::clamp(scale, 1, 8);
  uint32_t low_width = (width + scale - 1) / scale, low_height = (height + scale - 1) / scale;
  if (width != width_ || height != height_ || low_width != low_width_ || low_height != low_height_) {
    width_ = width;
    height_ = height;
    low_width_ = low_width;
    low_height_ = low_height;
    auto create = [&](uint32_t w, uint32_t h, std::unique_ptr<graphics::Image> &image) {
      graphics->CreateImage(w, h, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image);
    };
    create(width, height, visibility_);
    create(low_width, low_height, filtered_);
    graphics->CreateImage(width, height, graphics::IMAGE_FORMAT_D32_SFLOAT, &depth_);
    for (int i = 0; i < 2; ++i) {
      create(low_width, low_height, color_[i]);
      create(low_width, low_height, geometry_[i]);
      create(low_width, low_height, normal_[i]);
      commands->CmdClearImage(color_[i].get(), {});
      commands->CmdClearImage(geometry_[i].get(), {});
      commands->CmdClearImage(normal_[i].get(), {});
    }
    valid_ = false;
  }
  if (!visibility_program_ || buffer_count_ != buffers.size()) {
    const auto &vfs = core_->GetShadersVFS();
    if (graphics->CreateShader(vfs, "realtime/visibility.hlsl", "VSMain", "vs_6_0", {"-I."}, &vertex_) ||
        graphics->CreateShader(vfs, "realtime/visibility.hlsl", "PSMain", "ps_6_0", {"-I."}, &pixel_) ||
        graphics->CreateShader(vfs, "realtime/resolve.hlsl", "Main", "cs_6_0", {"-I."}, &resolve_shader_))
      throw std::runtime_error("failed to compile realtime visibility / reconstruction shaders");
    graphics->CreateProgram({graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT}, graphics::IMAGE_FORMAT_D32_SFLOAT,
                            &visibility_program_);
    visibility_program_->BindShader(vertex_.get(), graphics::SHADER_TYPE_VERTEX);
    visibility_program_->BindShader(pixel_.get(), graphics::SHADER_TYPE_PIXEL);
    visibility_program_->SetCullMode(graphics::CULL_MODE_NONE);
    visibility_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    visibility_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 2);
    visibility_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers.size());
    visibility_program_->Finalize();
    graphics->CreateComputeProgram(resolve_shader_.get(), &resolve_program_);
    resolve_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    resolve_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 2);
    resolve_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, buffers.size());
    resolve_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 4);
    resolve_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
    resolve_program_->Finalize();
    if (graphics->CreateShader(vfs, "realtime/filter.hlsl", "Main", "cs_6_0", {"-I."}, &filter_shader_))
      throw std::runtime_error("failed to compile realtime spatial filter");
    graphics->CreateComputeProgram(filter_shader_.get(), &filter_program_);
    filter_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    filter_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 3);
    filter_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
    filter_program_->Finalize();
    buffer_count_ = buffers.size();
    valid_ = false;
  }
  const auto &source = camera->Source();
  parameters_.previous_view_projection = parameters_.view_projection;
  parameters_.view_projection = glm::perspectiveZO(source.fovy, source.aspect, 0.1f, 100000.0f) * source.view;
  parameters_.camera_position = glm::inverse(source.view)[3];
  parameters_.extent = {width, height, low_width, low_height};
  float camera_change = 0;
  for (int column = 0; column < 4; ++column)
    for (int row = 0; row < 4; ++row)
      camera_change = std::max(camera_change, std::abs(parameters_.view_projection[column][row] -
                                                       parameters_.previous_view_projection[column][row]));
  if (camera_change > 1e-5f)
    history = std::min(history, 4);
  parameters_.config = {frame_index_++, valid_ && frame > 0 ? 1u : 0u, uint32_t(scale),
                        uint32_t(std::clamp(history, 1, 64) | (std::clamp(updates, 1, 16) << 8))};
  parameters_buffer_->UploadData(&parameters_, sizeof(parameters_));
  index_ = 1 - index_;
  auto counts = pipeline->VertexCounts();
  if (!draw_ranges_ || counts != vertex_counts_) {
    vertex_counts_ = counts;
    std::vector<uint32_t> ranges{static_cast<uint32_t>(counts.size())};
    uint64_t total = 0;
    for (auto count : counts) {
      total += count;
      if (total > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("realtime vertex count overflow");
      ranges.push_back(static_cast<uint32_t>(total));
    }
    vertex_count_ = static_cast<uint32_t>(total);
    graphics->CreateBuffer(ranges.size() * sizeof(uint32_t), graphics::BUFFER_TYPE_STATIC, &draw_ranges_);
    draw_ranges_->UploadData(ranges.data(), ranges.size() * sizeof(uint32_t));
  }
  graphics::GpuProfileScope visibility_profile(commands, "realtime_visibility");
  commands->CmdClearImage(visibility_.get(), {});
  commands->CmdClearImage(depth_.get(), {1, 0, 0, 0});
  commands->CmdBeginRendering({visibility_.get()}, depth_.get());
  commands->CmdBindProgram(visibility_program_.get());
  commands->CmdBindResources(0, {parameters_buffer_.get()}, graphics::BIND_POINT_GRAPHICS);
  commands->CmdBindResources(1, {pipeline->Instances(), draw_ranges_.get()}, graphics::BIND_POINT_GRAPHICS);
  commands->CmdBindResources(2, buffers, graphics::BIND_POINT_GRAPHICS);
  commands->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  commands->CmdSetViewport({0, 0, float(width), float(height), 0, 1});
  commands->CmdSetScissor({{0, 0}, {width, height}});
  if (vertex_count_)
    commands->CmdDraw(vertex_count_, 1, 0, 0);
  commands->CmdEndRendering();
  if (graphics::FrameProfile::active) {
    graphics::FrameProfile::active->counters["realtime_shading_pixels"] = low_width * low_height;
    graphics::FrameProfile::active->counters["realtime_software_gi"] = 1;
    graphics::FrameProfile::active->counters["realtime_max_shaded_pixels"] =
        ((low_width + std::clamp(updates, 1, 16) - 1) / std::clamp(updates, 1, 16)) * low_height;
  }
}

void RealtimeView::BindTrace(graphics::CommandContext *commands) {
  commands->CmdBindResources(0, {color_[index_].get(), geometry_[index_].get(), normal_[index_].get()},
                             graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(
      1, {visibility_.get(), color_[1 - index_].get(), geometry_[1 - index_].get(), normal_[1 - index_].get()},
      graphics::BIND_POINT_COMPUTE);
}

void RealtimeView::Resolve(graphics::CommandContext *commands,
                           graphics::Image *output,
                           render_shared::SoftwarePipeline *pipeline,
                           const std::vector<graphics::Buffer *> &buffers) {
  graphics::GpuProfileScope profile(commands, "realtime_reconstruct");
  commands->CmdBindComputeProgram(filter_program_.get());
  commands->CmdBindResources(0, {parameters_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(1, {color_[index_].get(), geometry_[index_].get(), normal_[index_].get()},
                             graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(2, {filtered_.get()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdDispatch((low_width_ + 7) / 8, (low_height_ + 7) / 8, 1);
  commands->CmdBindComputeProgram(resolve_program_.get());
  commands->CmdBindResources(0, {parameters_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(1, {pipeline->Instances(), draw_ranges_.get()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(2, buffers, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(3, {visibility_.get(), filtered_.get(), geometry_[index_].get(), normal_[index_].get()},
                             graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(4, {output}, graphics::BIND_POINT_COMPUTE);
  commands->CmdDispatch((width_ + 7) / 8, (height_ + 7) / 8, 1);
  valid_ = true;
}
}  // namespace sparkium::realtime
