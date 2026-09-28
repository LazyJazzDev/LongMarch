#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

namespace grassland::graphics::backend {
struct VulkanImGuiAssets {
  ImGuiContext *context;
  VkRenderPass render_pass{VK_NULL_HANDLE};
  VkFormat render_pass_format{VK_FORMAT_UNDEFINED};
  std::vector<VkFramebuffer> framebuffers;
  std::string font_path;
  float font_size;
  bool draw_command;
};
}  // namespace grassland::graphics::backend
