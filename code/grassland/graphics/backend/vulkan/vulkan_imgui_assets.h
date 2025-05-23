#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

namespace grassland::graphics::backend {
struct VulkanImGuiAssets {
  ImGuiContext *context;
  std::unique_ptr<vulkan::RenderPass> render_pass;
  std::vector<std::unique_ptr<vulkan::Framebuffer>> framebuffers;
  std::string font_path;
  float font_size;
  bool draw_command;
};
}  // namespace grassland::graphics::backend
