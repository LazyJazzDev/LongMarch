#include "grassland/graphics/backend/vulkan/vulkan_window.h"

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
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  vulkan::ThrowIfFailed(glfwCreateWindowSurface(core_->Instance(), GLFWWindow(), nullptr, &surface),
                        "Failed to create window surface");
  surface_ = std::make_unique<vulkan::Surface>(core_->Instance(), GLFWWindow(), surface);
  core_->Device()->CreateSwapchain(
      surface_.get(), enable_hdr_ ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM,
      enable_hdr_ ? VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, &swap_chain_);
  image_available_semaphores_.resize(swap_chain_->ImageCount());
  render_finish_semaphores_.resize(swap_chain_->ImageCount());
  for (size_t i = 0; i < image_available_semaphores_.size(); ++i) {
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vulkan::ThrowIfFailed(
        vkCreateSemaphore(core_->Device()->Handle(), &semaphore_info, nullptr, &image_available_semaphores_[i]),
        "Failed to create image available semaphore");
    vulkan::ThrowIfFailed(
        vkCreateSemaphore(core_->Device()->Handle(), &semaphore_info, nullptr, &render_finish_semaphores_[i]),
        "Failed to create render finish semaphore");
  }
  vkGetDeviceQueue(core_->Device()->Handle(), core_->Device()->PhysicalDevice().PresentFamilyIndex(surface_.get()), 0,
                   &present_queue_);
  ResizeEvent().RegisterCallback([this](int width, int height) { Rebuild(); });
}

VulkanWindow::~VulkanWindow() {
  if (GLFWWindow()) {
    VulkanWindow::CloseWindow();
  }
}

void VulkanWindow::CloseWindow() {
  core_->WaitGPU();
  if (imgui_assets_.context) {
    TerminateImGui();
  }
  for (VkSemaphore semaphore : image_available_semaphores_) {
    vkDestroySemaphore(core_->Device()->Handle(), semaphore, nullptr);
  }
  image_available_semaphores_.clear();
  for (VkSemaphore semaphore : render_finish_semaphores_) {
    vkDestroySemaphore(core_->Device()->Handle(), semaphore, nullptr);
  }
  render_finish_semaphores_.clear();
  swap_chain_.reset();
  surface_.reset();
  Window::CloseWindow();
}

vulkan::Swapchain *VulkanWindow::SwapChain() const {
  return swap_chain_.get();
}

VkSemaphore VulkanWindow::RenderFinishSemaphore() const {
  return render_finish_semaphores_[core_->CurrentFrame()];
}

VkSemaphore VulkanWindow::ImageAvailableSemaphore() const {
  return image_available_semaphores_[core_->CurrentFrame()];
}

uint32_t VulkanWindow::AcquireNextImage() {
  swap_chain_->AcquireNextImage(std::numeric_limits<uint64_t>::max(), ImageAvailableSemaphore(), VK_NULL_HANDLE,
                                &image_index_);
  return image_index_;
}

void VulkanWindow::Rebuild() {
  core_->WaitGPU();
  if (imgui_assets_.context) {
    DestroyImGuiFramebuffers();
  }
  swap_chain_.reset();
  core_->Device()->CreateSwapchain(
      surface_.get(), enable_hdr_ ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM,
      enable_hdr_ ? VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, &swap_chain_);
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    if (imgui_assets_.render_pass_format != swap_chain_->Format()) {
      ImGui_ImplVulkan_Shutdown();
      ImGui_ImplGlfw_Shutdown();

      ImGui::DestroyContext(imgui_assets_.context);
      DestroyImGuiRenderPass();
      SetupImGuiContext();
    }

    BuildImGuiFramebuffers();
  }
}

void VulkanWindow::Present() {
  VkSemaphore render_finish_semaphore = RenderFinishSemaphore();

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
  core_->WaitGPU();
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(imgui_assets_.context);
    imgui_assets_.context = nullptr;
  }
  DestroyImGuiFramebuffers();
  DestroyImGuiRenderPass();
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
  attachment_desc.format = swap_chain_->Format();
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
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &attachment_ref;
  VkRenderPassCreateInfo render_pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &attachment_desc;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  vulkan::ThrowIfFailed(
      vkCreateRenderPass(core_->Device()->Handle(), &render_pass_info, nullptr, &imgui_assets_.render_pass),
      "Failed to create ImGui render pass");
  imgui_assets_.render_pass_format = attachment_desc.format;

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.ApiVersion = VK_API_VERSION_1_2;
  init_info.Instance = core_->Instance();
  init_info.PhysicalDevice = core_->Device()->PhysicalDevice().Handle();
  init_info.Device = core_->Device()->Handle();
  init_info.QueueFamily = core_->GraphicsQueue()->QueueFamilyIndex();
  init_info.Queue = core_->GraphicsQueue()->Handle();
  init_info.DescriptorPoolSize = 32;
  init_info.RenderPass = imgui_assets_.render_pass;
  init_info.MinImageCount = 2;
  init_info.ImageCount = 3;
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
  imgui_assets_.framebuffers.resize(swap_chain_->ImageCount());
  for (int i = 0; i < swap_chain_->ImageCount(); i++) {
    VkImageView image_view = swap_chain_->ImageViews()[i];
    VkFramebufferCreateInfo framebuffer_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer_info.renderPass = imgui_assets_.render_pass;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = &image_view;
    framebuffer_info.width = swap_chain_->Extent().width;
    framebuffer_info.height = swap_chain_->Extent().height;
    framebuffer_info.layers = 1;
    vulkan::ThrowIfFailed(
        vkCreateFramebuffer(core_->Device()->Handle(), &framebuffer_info, nullptr, &imgui_assets_.framebuffers[i]),
        "Failed to create ImGui framebuffer");
  }
}

void VulkanWindow::DestroyImGuiFramebuffers() {
  for (VkFramebuffer framebuffer : imgui_assets_.framebuffers) {
    vkDestroyFramebuffer(core_->Device()->Handle(), framebuffer, nullptr);
  }
  imgui_assets_.framebuffers.clear();
}

void VulkanWindow::DestroyImGuiRenderPass() {
  if (imgui_assets_.render_pass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(core_->Device()->Handle(), imgui_assets_.render_pass, nullptr);
    imgui_assets_.render_pass = VK_NULL_HANDLE;
    imgui_assets_.render_pass_format = VK_FORMAT_UNDEFINED;
  }
}

}  // namespace grassland::graphics::backend
