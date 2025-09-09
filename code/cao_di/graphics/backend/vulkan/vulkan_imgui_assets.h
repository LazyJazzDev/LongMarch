#pragma once
#include "cao_di/graphics/backend/vulkan/vulkan_core.h"
#include "cao_di/graphics/backend/vulkan/vulkan_util.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

namespace CD::graphics::backend {
struct VulkanImGuiAssets {
  ImGuiContext *context;
  std::unique_ptr<vulkan::RenderPass> render_pass;
  std::vector<std::unique_ptr<vulkan::Framebuffer>> framebuffers;
  std::string font_path;
  float font_size;
  bool draw_command;
};
}  // namespace CD::graphics::backend
