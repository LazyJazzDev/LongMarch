#include "grassland/graphics/backend/vulkan/vulkan_window.h"

#include "grassland/graphics/backend/vulkan/vulkan_image.h"

namespace grassland::graphics::backend {

VulkanWindow::VulkanWindow(VulkanCore *core,
                           int width,
                           int height,
                           const std::string &title,
                           bool fullscreen,
                           bool resizable,
                           bool enable_hdr)
    : Window(width, height, title, fullscreen, resizable, enable_hdr),
      core_(core) {
  core_->Instance()->CreateSurfaceFromGLFWWindow(GLFWWindow(), &surface_);
  surface_format_ = SelectSurfaceFormat(enable_hdr_);
  if (core_->Device()->CreateSwapchain(surface_.get(), surface_format_.format, surface_format_.colorSpace,
                                       &swap_chain_) != VK_SUCCESS)
    throw std::runtime_error("Failed to create Vulkan swapchain");
  image_available_semaphores_.resize(swap_chain_->ImageCount());
  render_finish_semaphores_.resize(swap_chain_->ImageCount());
  for (size_t i = 0; i < image_available_semaphores_.size(); ++i) {
    core_->Device()->CreateSemaphore(&image_available_semaphores_[i]);
    core_->Device()->CreateSemaphore(&render_finish_semaphores_[i]);
  }
  vkGetDeviceQueue(core_->Device()->Handle(), core_->Device()->PhysicalDevice().PresentFamilyIndex(surface_.get()), 0,
                   &present_queue_);
  // Swapchains follow pixels, not logical window coordinates. In particular,
  // GLFW on Wayland emits only a framebuffer callback for programmatic resizes
  // and for fractional-scale changes without a logical size change.
  FramebufferResizeEvent().RegisterCallback([this](int width, int height) {
    if (width > 0 && height > 0)
      Rebuild();
  });
}

VulkanWindow::~VulkanWindow() {
  if (GLFWWindow()) {
    VulkanWindow::CloseWindow();
  }
}

void VulkanWindow::CloseWindow() {
  core_->WaitGPU();
  hdr_framebuffer_.reset();
  if (imgui_assets_.context) {
    TerminateImGui();
  }
  image_available_semaphores_.clear();
  render_finish_semaphores_.clear();
  swap_chain_.reset();
  surface_.reset();
  Window::CloseWindow();
}

void VulkanWindow::SetHDR(bool enable_hdr) {
  if (enable_hdr == enable_hdr_)
    return;
  // Probe before changing state so an unsupported request leaves SDR usable.
  SelectSurfaceFormat(enable_hdr);
  const bool previous = enable_hdr_;
  try {
    Window::SetHDR(enable_hdr);
    // A format change needs a rebuild even when the framebuffer size is fixed.
    Rebuild();
  } catch (...) {
    enable_hdr_ = previous;
    throw;
  }
}

VkSurfaceFormatKHR VulkanWindow::ChooseHDRSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) {
  for (const auto &candidate :
       {VkSurfaceFormatKHR{VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT},
        VkSurfaceFormatKHR{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT},
        VkSurfaceFormatKHR{VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT}}) {
    for (const auto &format : formats) {
      if (format.format == candidate.format && format.colorSpace == candidate.colorSpace)
        return format;
    }
  }
  throw std::runtime_error(
      "The Vulkan surface supports neither linear scRGB nor HDR10/PQ; "
      "use an HDR-capable desktop and native Wayland on Linux (X11 remains available for SDR)");
}

VkSurfaceFormatKHR VulkanWindow::SelectSurfaceFormat(bool hdr) const {
  const auto support =
      vulkan::Swapchain::QuerySwapChainSupport(core_->Device()->PhysicalDevice().Handle(), surface_->Handle());
  auto format = hdr ? ChooseHDRSurfaceFormat(support.formats)
                    : vulkan::Swapchain::ChooseSwapSurfaceFormat(support.formats, VK_FORMAT_R8G8B8A8_UNORM,
                                                                 VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
  VkFormatProperties properties{};
  vkGetPhysicalDeviceFormatProperties(core_->Device()->PhysicalDevice().Handle(), format.format, &properties);
  if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
    throw std::runtime_error("Vulkan presentation format does not support image blits");
  return format;
}

bool VulkanWindow::UsesPQOutput() const {
  return enable_hdr_ && surface_format_.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT;
}

VkFormat VulkanWindow::ImGuiFormat() const {
  return UsesPQOutput() ? VK_FORMAT_R16G16B16A16_SFLOAT : swap_chain_->Format();
}

vulkan::Swapchain *VulkanWindow::SwapChain() const {
  return swap_chain_.get();
}

vulkan::Semaphore *VulkanWindow::RenderFinishSemaphore() const {
  return render_finish_semaphores_[core_->CurrentFrame()].get();
}

vulkan::Semaphore *VulkanWindow::ImageAvailableSemaphore() const {
  return image_available_semaphores_[core_->CurrentFrame()].get();
}

uint32_t VulkanWindow::AcquireNextImage() {
  swap_chain_->AcquireNextImage(std::numeric_limits<uint64_t>::max(),
                                image_available_semaphores_[core_->CurrentFrame()]->Handle(), VK_NULL_HANDLE,
                                &image_index_);
  return image_index_;
}

void VulkanWindow::Rebuild() {
  core_->WaitGPU();
  const auto format = SelectSurfaceFormat(enable_hdr_);
  std::unique_ptr<vulkan::Swapchain> next;
  if (core_->Device()->CreateSwapchain(surface_.get(), format.format, format.colorSpace, &next,
                                       swap_chain_->Handle()) != VK_SUCCESS)
    throw std::runtime_error("Failed to rebuild Vulkan swapchain");
  hdr_framebuffer_.reset();
  imgui_assets_.framebuffers.clear();
  swap_chain_ = std::move(next);
  surface_format_ = format;
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    imgui_assets_.framebuffers.clear();
    if (imgui_assets_.render_pass->AttachmentDescriptions()[0].format != ImGuiFormat()) {
      ImGui_ImplVulkan_Shutdown();
      ImGui_ImplGlfw_Shutdown();

      ImGui::DestroyContext(imgui_assets_.context);
      imgui_assets_.render_pass.reset();
      SetupImGuiContext();
    }

    BuildImGuiFramebuffers();
  }
}

void VulkanWindow::Present() {
  VkSemaphore render_finish_semaphore = render_finish_semaphores_[core_->CurrentFrame()]->Handle();

  VkSwapchainKHR swap_chain = swap_chain_->Handle();

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &render_finish_semaphore;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swap_chain;
  presentInfo.pImageIndices = &image_index_;

  auto result = vkQueuePresentKHR(present_queue_, &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    Rebuild();
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to present swap chain image");
  }
}

void VulkanWindow::InitImGui(const char *font_file_path, float font_size) {
  imgui_assets_.font_size = font_size;
  if (font_file_path) {
    imgui_assets_.font_path = font_file_path;
  }

  SetupImGuiContext();
  BuildImGuiFramebuffers();
}

void VulkanWindow::TerminateImGui() {
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(imgui_assets_.context);
    imgui_assets_.context = nullptr;
  }
}

void VulkanWindow::BeginImGuiFrame() {
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
  }
}

void VulkanWindow::EndImGuiFrame() {
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    ImGui::Render();
    imgui_assets_.draw_command = true;
  }
}

ImGuiContext *VulkanWindow::GetImGuiContext() const {
  return imgui_assets_.context;
}

VulkanImGuiAssets &VulkanWindow::ImGuiAssets() {
  return imgui_assets_;
}

void VulkanWindow::SetupImGuiContext() {
  imgui_assets_.context = ImGui::CreateContext();
  ImGui::SetCurrentContext(imgui_assets_.context);
  ImGui::StyleColorsClassic();
  ImGui_ImplGlfw_InitForVulkan(GLFWWindow(), true);

  VkAttachmentDescription attachment_desc{};
  attachment_desc.flags = 0;
  attachment_desc.format = ImGuiFormat();
  attachment_desc.samples = VK_SAMPLE_COUNT_1_BIT;
  attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachment_desc.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
  attachment_desc.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
  VkAttachmentReference attachment_ref{};
  attachment_ref.attachment = 0;
  attachment_ref.layout = VK_IMAGE_LAYOUT_GENERAL;
  core_->Device()->CreateRenderPass({attachment_desc}, {attachment_ref}, &imgui_assets_.render_pass);

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.ApiVersion = VK_API_VERSION_1_2;
  init_info.Instance = core_->Instance()->Handle();
  init_info.PhysicalDevice = core_->Device()->PhysicalDevice().Handle();
  init_info.Device = core_->Device()->Handle();
  init_info.QueueFamily = core_->GraphicsQueue()->QueueFamilyIndex();
  init_info.Queue = core_->GraphicsQueue()->Handle();
  init_info.DescriptorPoolSize = 32;
  init_info.RenderPass = imgui_assets_.render_pass->Handle();
  init_info.MinImageCount = 2;
  init_info.ImageCount = swap_chain_->ImageCount();
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&init_info);

  auto &io = ImGui::GetIO();
  if (!imgui_assets_.font_path.empty()) {
    io.Fonts->AddFontFromFileTTF(imgui_assets_.font_path.c_str(), imgui_assets_.font_size, nullptr,
                                 io.Fonts->GetGlyphRangesChineseFull());
    io.Fonts->Build();
  } else {
    ImFontConfig im_font_config{};
    im_font_config.SizePixels = imgui_assets_.font_size;
    io.Fonts->AddFontDefault(&im_font_config);
  }

  ImGui_ImplVulkan_CreateFontsTexture();
  imgui_assets_.draw_command = false;
}

void VulkanWindow::BuildImGuiFramebuffers() {
  // PQ is encoded only after scene/UI blending in the floating-point target.
  if (UsesPQOutput())
    return;
  imgui_assets_.framebuffers.resize(swap_chain_->ImageCount());
  for (int i = 0; i < swap_chain_->ImageCount(); i++) {
    imgui_assets_.render_pass->CreateFramebuffer({swap_chain_->ImageViews()[i]}, swap_chain_->Extent(),
                                                 &imgui_assets_.framebuffers[i]);
  }
}

vulkan::Framebuffer *VulkanWindow::HDRFramebuffer(VulkanImage *image) {
  if (!hdr_framebuffer_)
    imgui_assets_.render_pass->CreateFramebuffer({image->Image()->ImageView()}, image->Image()->Extent(),
                                                 &hdr_framebuffer_);
  return hdr_framebuffer_.get();
}

}  // namespace grassland::graphics::backend
