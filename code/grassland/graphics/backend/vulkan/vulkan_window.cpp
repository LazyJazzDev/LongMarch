#include "grassland/graphics/backend/vulkan/vulkan_window.h"

#include "grassland/graphics/backend/vulkan/helper/swap_chain.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"

namespace grassland::graphics::backend {

void VulkanWindow::DestroySwapChain() {
  for (VkImageView view : swap_chain_image_views_) {
    vkDestroyImageView(core_->Handle(), view, nullptr);
  }
  swap_chain_image_views_.clear();
  swap_chain_images_.clear();
  if (swap_chain_) {
    vkDestroySwapchainKHR(core_->Handle(), swap_chain_, nullptr);
    swap_chain_ = VK_NULL_HANDLE;
  }
}

VkResult VulkanWindow::CreatePresentationSwapchain(VkSurfaceFormatKHR surface_format,
                                                   VkSwapchainKHR *result,
                                                   VkSwapchainKHR old_swapchain) {
  VkDevice device = core_->Handle();
  auto physical_device = core_->PhysicalDevice();
  auto support = vulkan::QuerySwapChainSupport(physical_device, surface_);
  auto present_mode = vulkan::ChooseSwapPresentMode(support.presentModes);
  swap_chain_extent_ = vulkan::ChooseSwapExtent(support.capabilities, GLFWWindow());
  uint32_t image_count = support.capabilities.minImageCount + 1;
  if (support.capabilities.maxImageCount && image_count > support.capabilities.maxImageCount) {
    image_count = support.capabilities.maxImageCount;
  }
  VkSwapchainCreateInfoKHR create_info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  create_info.surface = surface_;
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = swap_chain_extent_;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  uint32_t families[] = {vulkan::GraphicsFamilyIndex(physical_device),
                         vulkan::PresentFamilyIndex(physical_device, surface_)};
  if (families[0] != families[1]) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = families;
  } else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }
  create_info.preTransform = support.capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = old_swapchain;
  return vkCreateSwapchainKHR(device, &create_info, nullptr, result);
}

void VulkanWindow::LoadSwapChainImages() {
  VkDevice device = core_->Handle();
  uint32_t image_count = 0;
  vulkan::ThrowIfFailed(vkGetSwapchainImagesKHR(device, swap_chain_, &image_count, nullptr),
                        "Failed to query Vulkan swap chain images");
  swap_chain_images_.resize(image_count);
  vulkan::ThrowIfFailed(vkGetSwapchainImagesKHR(device, swap_chain_, &image_count, swap_chain_images_.data()),
                        "Failed to get Vulkan swap chain images");
  swap_chain_image_views_.resize(image_count);
  for (size_t i = 0; i < image_count; ++i) {
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = swap_chain_images_[i];
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = swap_chain_format_;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    vulkan::ThrowIfFailed(vkCreateImageView(device, &view_info, nullptr, &swap_chain_image_views_[i]),
                          "Failed to create Vulkan swap chain image view");
  }
}

void VulkanWindow::CreateSwapChain() {
  const auto format = SelectSurfaceFormat(enable_hdr_);
  if (!format)
    throw std::runtime_error("No supported Vulkan presentation format");
  vulkan::ThrowIfFailed(CreatePresentationSwapchain(*format, &swap_chain_, VK_NULL_HANDLE),
                        "Failed to create Vulkan swap chain");
  surface_format_ = *format;
  swap_chain_format_ = format->format;
  LoadSwapChainImages();
}

VulkanWindow::VulkanWindow(VulkanCore *core,
                           int width,
                           int height,
                           const std::string &title,
                           bool fullscreen,
                           bool resizable,
                           bool enable_hdr)
    : Window(width, height, title, fullscreen, resizable, enable_hdr),
      core_(core) {
  vulkan::ThrowIfFailed(glfwCreateWindowSurface(core_->Instance(), GLFWWindow(), nullptr, &surface_),
                        "Failed to create window surface");
  CreateSwapChain();
  image_available_semaphores_.resize(swap_chain_images_.size());
  render_finish_semaphores_.resize(swap_chain_images_.size());
  for (size_t i = 0; i < image_available_semaphores_.size(); ++i) {
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vulkan::ThrowIfFailed(vkCreateSemaphore(core_->Handle(), &semaphore_info, nullptr, &image_available_semaphores_[i]),
                          "Failed to create image available semaphore");
    vulkan::ThrowIfFailed(vkCreateSemaphore(core_->Handle(), &semaphore_info, nullptr, &render_finish_semaphores_[i]),
                          "Failed to create render finish semaphore");
  }
  vkGetDeviceQueue(core_->Handle(), vulkan::PresentFamilyIndex(core_->PhysicalDevice(), surface_), 0, &present_queue_);
  FramebufferResizeEvent().RegisterCallback([this](int width, int height) {
    if (width > 0 && height > 0 && Rebuild() != 0)
      LogError("Failed to resize Vulkan presentation");
  });
}

VulkanWindow::~VulkanWindow() {
  if (GLFWWindow()) {
    VulkanWindow::CloseWindow();
  }
}

void VulkanWindow::CloseWindow() {
  core_->WaitGPU();
  if (hdr_framebuffer_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(core_->Handle(), hdr_framebuffer_, nullptr);
    hdr_framebuffer_ = VK_NULL_HANDLE;
  }
  if (imgui_assets_.context) {
    TerminateImGui();
  }
  for (VkSemaphore semaphore : image_available_semaphores_) {
    vkDestroySemaphore(core_->Handle(), semaphore, nullptr);
  }
  image_available_semaphores_.clear();
  for (VkSemaphore semaphore : render_finish_semaphores_) {
    vkDestroySemaphore(core_->Handle(), semaphore, nullptr);
  }
  render_finish_semaphores_.clear();
  DestroySwapChain();
  vkDestroySurfaceKHR(core_->Instance(), surface_, nullptr);
  surface_ = VK_NULL_HANDLE;
  Window::CloseWindow();
}

int VulkanWindow::SetHDR(bool enable_hdr) {
  if (!GLFWWindow() || presentation_failed_)
    return -1;
  if (enable_hdr == enable_hdr_)
    return 0;
  if (!SelectSurfaceFormat(enable_hdr)) {
    LogWarning("Requested Vulkan HDR/SDR presentation mode is unavailable");
    return -1;
  }
  const bool previous = enable_hdr_;
  enable_hdr_ = enable_hdr;
  try {
    if (Rebuild() != 0) {
      enable_hdr_ = previous;
      return -1;
    }
    return Window::SetHDR(enable_hdr);
  } catch (const std::exception &error) {
    enable_hdr_ = previous;
    LogError("Failed to change Vulkan HDR presentation: {}", error.what());
    return -1;
  }
}

std::optional<VkSurfaceFormatKHR> VulkanWindow::ChooseHDRSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) {
  for (const auto &candidate :
       {VkSurfaceFormatKHR{VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT},
        VkSurfaceFormatKHR{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT},
        VkSurfaceFormatKHR{VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT}}) {
    for (const auto &format : formats) {
      if (format.format == candidate.format && format.colorSpace == candidate.colorSpace)
        return format;
    }
  }
  return std::nullopt;
}

std::optional<VkSurfaceFormatKHR> VulkanWindow::SelectSurfaceFormat(bool hdr) const {
  const auto support = vulkan::QuerySwapChainSupport(core_->PhysicalDevice(), surface_);
  if (support.formats.empty())
    return std::nullopt;
  auto format = hdr ? ChooseHDRSurfaceFormat(support.formats)
                    : std::optional<VkSurfaceFormatKHR>(vulkan::ChooseSwapSurfaceFormat(
                          support.formats, VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR));
  if (!format || (!hdr && format->colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR))
    return std::nullopt;
  VkFormatProperties properties{};
  vkGetPhysicalDeviceFormatProperties(core_->PhysicalDevice(), format->format, &properties);
  if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
    return std::nullopt;
  return format;
}

bool VulkanWindow::UsesPQOutput() const {
  return enable_hdr_ && surface_format_.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT;
}

VkFormat VulkanWindow::ImGuiFormat() const {
  return UsesPQOutput() ? VK_FORMAT_R16G16B16A16_SFLOAT : swap_chain_format_;
}

VkSemaphore VulkanWindow::RenderFinishSemaphore() const {
  return render_finish_semaphores_[core_->CurrentFrame()];
}

VkSemaphore VulkanWindow::ImageAvailableSemaphore() const {
  return image_available_semaphores_[core_->CurrentFrame()];
}

uint32_t VulkanWindow::AcquireNextImage() {
  if (presentation_failed_)
    throw std::runtime_error("Vulkan presentation unavailable after swapchain recovery failed");
  vulkan::ThrowIfFailed(vkAcquireNextImageKHR(core_->Handle(), swap_chain_, std::numeric_limits<uint64_t>::max(),
                                              ImageAvailableSemaphore(), VK_NULL_HANDLE, &image_index_),
                        "Failed to acquire Vulkan swapchain image");
  return image_index_;
}

int VulkanWindow::Rebuild() {
  if (presentation_failed_)
    return -1;
  core_->WaitGPU();
  const auto format = SelectSurfaceFormat(enable_hdr_);
  if (!format)
    return -1;
  const auto previous_format = surface_format_;
  const bool previous_hdr = previous_format.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  VkSwapchainKHR next = VK_NULL_HANDLE;
  const auto status = CreatePresentationSwapchain(*format, &next, swap_chain_);
  if (hdr_framebuffer_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(core_->Handle(), hdr_framebuffer_, nullptr);
    hdr_framebuffer_ = VK_NULL_HANDLE;
  }
  if (imgui_assets_.context) {
    DestroyImGuiFramebuffers();
  }
  DestroySwapChain();
  bool recovered = false;
  if (status != VK_SUCCESS) {
    if (next != VK_NULL_HANDLE)
      vkDestroySwapchainKHR(core_->Handle(), next, nullptr);
    enable_hdr_ = previous_hdr;
    if (CreatePresentationSwapchain(previous_format, &next, VK_NULL_HANDLE) != VK_SUCCESS) {
      presentation_failed_ = true;
      LogError("Vulkan presentation disabled: could not recreate the previous swapchain mode");
      return -1;
    }
    recovered = true;
  }
  swap_chain_ = next;
  surface_format_ = recovered ? previous_format : *format;
  swap_chain_format_ = surface_format_.format;
  LoadSwapChainImages();
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    if (imgui_assets_.render_pass_format != ImGuiFormat()) {
      ImGui_ImplVulkan_Shutdown();
      ImGui_ImplGlfw_Shutdown();

      ImGui::DestroyContext(imgui_assets_.context);
      DestroyImGuiRenderPass();
      SetupImGuiContext();
    }

    BuildImGuiFramebuffers();
  }
  return recovered ? -1 : 0;
}

void VulkanWindow::Present() {
  if (presentation_failed_)
    throw std::runtime_error("Vulkan presentation unavailable after swapchain recovery failed");
  VkSemaphore render_finish_semaphore = RenderFinishSemaphore();

  VkSwapchainKHR swap_chain = swap_chain_;

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &render_finish_semaphore;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swap_chain;
  presentInfo.pImageIndices = &image_index_;

  auto result = vkQueuePresentKHR(present_queue_, &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    if (Rebuild() != 0)
      throw std::runtime_error("Failed to rebuild Vulkan swapchain after presentation");
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
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &attachment_ref;
  VkRenderPassCreateInfo render_pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &attachment_desc;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  vulkan::ThrowIfFailed(vkCreateRenderPass(core_->Handle(), &render_pass_info, nullptr, &imgui_assets_.render_pass),
                        "Failed to create ImGui render pass");
  imgui_assets_.render_pass_format = attachment_desc.format;

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.ApiVersion = VK_API_VERSION_1_2;
  init_info.Instance = core_->Instance();
  init_info.PhysicalDevice = core_->PhysicalDevice();
  init_info.Device = core_->Handle();
  init_info.QueueFamily = vulkan::GraphicsFamilyIndex(core_->PhysicalDevice());
  init_info.Queue = core_->GraphicsQueue();
  init_info.DescriptorPoolSize = 32;
  init_info.RenderPass = imgui_assets_.render_pass;
  init_info.MinImageCount = 2;
  init_info.ImageCount = static_cast<uint32_t>(swap_chain_images_.size());
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
  if (UsesPQOutput())
    return;
  imgui_assets_.framebuffers.resize(swap_chain_images_.size());
  for (int i = 0; i < swap_chain_images_.size(); i++) {
    VkImageView image_view = swap_chain_image_views_[i];
    VkFramebufferCreateInfo framebuffer_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer_info.renderPass = imgui_assets_.render_pass;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = &image_view;
    framebuffer_info.width = swap_chain_extent_.width;
    framebuffer_info.height = swap_chain_extent_.height;
    framebuffer_info.layers = 1;
    vulkan::ThrowIfFailed(
        vkCreateFramebuffer(core_->Handle(), &framebuffer_info, nullptr, &imgui_assets_.framebuffers[i]),
        "Failed to create ImGui framebuffer");
  }
}

void VulkanWindow::DestroyImGuiFramebuffers() {
  for (VkFramebuffer framebuffer : imgui_assets_.framebuffers) {
    vkDestroyFramebuffer(core_->Handle(), framebuffer, nullptr);
  }
  imgui_assets_.framebuffers.clear();
}

void VulkanWindow::DestroyImGuiRenderPass() {
  if (imgui_assets_.render_pass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(core_->Handle(), imgui_assets_.render_pass, nullptr);
    imgui_assets_.render_pass = VK_NULL_HANDLE;
    imgui_assets_.render_pass_format = VK_FORMAT_UNDEFINED;
  }
}

VkFramebuffer VulkanWindow::HDRFramebuffer(VulkanImage *image) {
  if (hdr_framebuffer_ == VK_NULL_HANDLE) {
    VkImageView image_view = image->ImageView();
    const auto extent = image->Extent();
    VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    info.renderPass = imgui_assets_.render_pass;
    info.attachmentCount = 1;
    info.pAttachments = &image_view;
    info.width = extent.width;
    info.height = extent.height;
    info.layers = 1;
    vulkan::ThrowIfFailed(vkCreateFramebuffer(core_->Handle(), &info, nullptr, &hdr_framebuffer_),
                          "Failed to create HDR ImGui framebuffer");
  }
  return hdr_framebuffer_;
}

}  // namespace grassland::graphics::backend
