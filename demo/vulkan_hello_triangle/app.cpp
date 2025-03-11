#include "app.h"

namespace {

#include "built_in_shaders.inl"

}

Application::Application() {
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(800, 600, "Vulkan", nullptr, nullptr);
  if (!window_) {
    throw std::runtime_error("Failed to create window");
  }
}

Application::~Application() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void Application::Run() {
  OnInit();
  while (!glfwWindowShouldClose(window_)) {
    OnUpdate();
    OnRender();
    glfwPollEvents();
  }
  device_->WaitIdle();
  OnShutdown();
}

void Application::OnInit() {
  VkResult result;
  result = long_march::vulkan::CreateInstance(long_march::vulkan::InstanceCreateHint{}, &instance_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance");
  }

  result = instance_->CreateSurfaceFromGLFWWindow(window_, &surface_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan surface");
  }

  long_march::vulkan::DeviceFeatureRequirement feature_requirement;
  feature_requirement.enable_raytracing_extension = false;

  result = instance_->CreateDevice(surface_.get(), false, -1, &device_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan device");
  }

  long_march::LogInfo("Device: {}", device_->PhysicalDevice().GetPhysicalDeviceProperties().deviceName);

  result = device_->CreateSwapchain(surface_.get(), &swapchain_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan swapchain");
  }

  device_->GetQueue(device_->PhysicalDevice().GraphicsFamilyIndex(), 0, &graphics_queue_);
  device_->GetQueue(device_->PhysicalDevice().PresentFamilyIndex(surface_.get()), 0, &present_queue_);
  device_->GetQueue(device_->PhysicalDevice().TransferFamilyIndex(), -1, &transfer_queue_);

  result = device_->CreateCommandPool(&graphics_command_pool_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create graphics command pool");
  }

  result = device_->CreateCommandPool(device_->PhysicalDevice().TransferFamilyIndex(),
                                      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, &transfer_command_pool_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create transfer command pool");
  }

  command_buffers_.resize(max_frames_in_flight_);

  image_available_semaphores_.resize(max_frames_in_flight_);
  render_finished_semaphores_.resize(max_frames_in_flight_);
  in_flight_fences_.resize(max_frames_in_flight_);

  for (int i = 0; i < max_frames_in_flight_; ++i) {
    result = device_->CreateSemaphore(&image_available_semaphores_[i]);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create image available semaphore");
    }
    result = device_->CreateSemaphore(&render_finished_semaphores_[i]);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create render finished semaphore");
    }
    result = device_->CreateFence(true, &in_flight_fences_[i]);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create in flight fence");
    }
    result = graphics_command_pool_->AllocateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, &command_buffers_[i]);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate command buffer");
    }
  }

  result = device_->CreateShaderModule(
      long_march::vulkan::CompileGLSLToSPIRV(GetShaderCode("shaders/hello_triangle.vert"), VK_SHADER_STAGE_VERTEX_BIT),
      &vertex_shader_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create vertex shader module");
  }

  result = device_->CreateShaderModule(long_march::vulkan::CompileGLSLToSPIRV(
                                           GetShaderCode("shaders/hello_triangle.frag"), VK_SHADER_STAGE_FRAGMENT_BIT),
                                       &fragment_shader_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create fragment shader module");
  }

  result = device_->CreateImage(VK_FORMAT_B8G8R8A8_UNORM, swapchain_->Extent(), &frame_image_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create frame image");
  }

  std::vector<VkAttachmentDescription> attachments(1);
  std::vector<VkAttachmentReference> color_references(1);

  attachments[0].format = VK_FORMAT_B8G8R8A8_UNORM;
  attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

  color_references[0].attachment = 0;
  color_references[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  result = device_->CreateRenderPass(attachments, color_references, &render_pass_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create render pass");
  }

  std::vector<VkImageView> attachments_view = {frame_image_->ImageView()};
  result = render_pass_->CreateFramebuffer(attachments_view, swapchain_->Extent(), &framebuffer_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create framebuffer");
  }

  std::vector<Vertex> vertices = {
      {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
      {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
      {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
  };

  std::vector<uint32_t> indices = {0, 1, 2};

  result = device_->CreateBuffer(vertices.size() * sizeof(Vertex),
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_MEMORY_USAGE_GPU_ONLY, &vertex_buffer_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create vertex buffer");
  }

  result = device_->CreateBuffer(indices.size() * sizeof(uint32_t),
                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_MEMORY_USAGE_GPU_ONLY, &index_buffer_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create index buffer");
  }

  long_march::vulkan::UploadBuffer(transfer_queue_.get(), transfer_command_pool_.get(), vertex_buffer_.get(),
                                   vertices.data(), vertices.size() * sizeof(Vertex));
  long_march::vulkan::UploadBuffer(transfer_queue_.get(), transfer_command_pool_.get(), index_buffer_.get(),
                                   indices.data(), indices.size() * sizeof(uint32_t));

  result = device_->CreatePipelineLayout({}, &pipeline_layout_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout");
  }

  long_march::vulkan::PipelineSettings pipeline_settings(render_pass_.get(), pipeline_layout_.get(), 0);
  pipeline_settings.AddInputBinding(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX);
  pipeline_settings.AddInputAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos));
  pipeline_settings.AddInputAttribute(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color));
  pipeline_settings.AddShaderStage(vertex_shader_.get(), VK_SHADER_STAGE_VERTEX_BIT);
  pipeline_settings.AddShaderStage(fragment_shader_.get(), VK_SHADER_STAGE_FRAGMENT_BIT);
  pipeline_settings.SetPrimitiveTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  pipeline_settings.SetMultiSampleState(VK_SAMPLE_COUNT_1_BIT);
  pipeline_settings.SetCullMode(VK_CULL_MODE_BACK_BIT);

  result = device_->CreatePipeline(pipeline_settings, &pipeline_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline");
  }
}

void Application::OnShutdown() {
  pipeline_.reset();
  pipeline_layout_.reset();

  vertex_buffer_.reset();
  index_buffer_.reset();

  framebuffer_.reset();
  render_pass_.reset();
  frame_image_.reset();
  fragment_shader_.reset();
  vertex_shader_.reset();
  image_available_semaphores_.clear();
  render_finished_semaphores_.clear();
  in_flight_fences_.clear();
  command_buffers_.clear();
  graphics_command_pool_.reset();
  transfer_command_pool_.reset();
  graphics_queue_.reset();
  present_queue_.reset();
  transfer_queue_.reset();
  swapchain_.reset();
  device_.reset();
  surface_.reset();
  instance_.reset();
}

void Application::OnUpdate() {
}

void Application::OnRender() {
  BeginFrame();

  VkImage swapchain_image = swapchain_->Image(image_index_);

  VkCommandBuffer command_buffer = command_buffers_[current_frame_]->Handle();

  // Begin RenderPass
  VkRenderPassBeginInfo render_pass_begin_info{};
  render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_begin_info.renderPass = render_pass_->Handle();
  render_pass_begin_info.framebuffer = framebuffer_->Handle();
  render_pass_begin_info.renderArea.offset = {0, 0};
  render_pass_begin_info.renderArea.extent = swapchain_->Extent();
  VkClearValue clear_color = {0.6f, 0.7f, 0.8f, 1.0f};
  render_pass_begin_info.clearValueCount = 1;
  render_pass_begin_info.pClearValues = &clear_color;

  vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

  // Bind Pipeline
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->Handle());

  // Set Scissor and Viewport
  VkExtent2D extent = swapchain_->Extent();
  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(extent.width);
  viewport.height = static_cast<float>(extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(command_buffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = extent;
  vkCmdSetScissor(command_buffer, 0, 1, &scissor);

  // Bind Vertex Buffer
  VkBuffer vertex_buffers[] = {vertex_buffer_->Handle()};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);

  // Bind Index Buffer
  vkCmdBindIndexBuffer(command_buffer, index_buffer_->Handle(), 0, VK_INDEX_TYPE_UINT32);

  // Draw
  vkCmdDrawIndexed(command_buffer, 3, 1, 0, 0, 0);

  // End RenderPass
  vkCmdEndRenderPass(command_buffer);

  // Transition swapchain_image layout from UNDEFINED to TRANSFER_DST
  long_march::vulkan::TransitImageLayout(command_buffer, swapchain_image, VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                         VK_IMAGE_ASPECT_COLOR_BIT);

  // Copy from frame_image to swapchain_image
  VkImageCopy copy_region{};
  copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.srcSubresource.layerCount = 1;
  copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.dstSubresource.layerCount = 1;
  copy_region.extent.width = swapchain_->Extent().width;
  copy_region.extent.height = swapchain_->Extent().height;
  copy_region.extent.depth = 1;

  vkCmdCopyImage(command_buffer, frame_image_->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_image,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

  // Transition swapchain_image layout from TRANSFER_DST to PRESENT_SRC
  long_march::vulkan::TransitImageLayout(command_buffer, swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                                         VK_IMAGE_ASPECT_COLOR_BIT);

  EndFrame();
}

void Application::BeginFrame() {
  VkFence fence = in_flight_fences_[current_frame_]->Handle();
  vkWaitForFences(device_->Handle(), 1, &fence, VK_TRUE, UINT64_MAX);
  vkResetFences(device_->Handle(), 1, &fence);

  if (window_) {
    VkSemaphore image_available_semaphore = image_available_semaphores_[current_frame_]->Handle();
    VkResult result = swapchain_->AcquireNextImage(std::numeric_limits<uint64_t>::max(), image_available_semaphore,
                                                   VK_NULL_HANDLE, &image_index_);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      // Recreate swapchain
      device_->WaitIdle();
      swapchain_.reset();
      device_->CreateSwapchain(surface_.get(), &swapchain_);
      return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("Failed to acquire next image");
    }
  }

  VkCommandBuffer command_buffer = command_buffers_[current_frame_]->Handle();

  vkResetCommandBuffer(command_buffer, 0);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
    throw std::runtime_error("Failed to begin recording command buffer");
  }
}

void Application::EndFrame() {
  VkCommandBuffer command_buffer = command_buffers_[current_frame_]->Handle();
  if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
    throw std::runtime_error("Failed to record command buffer");
  }

  VkSemaphore image_available_semaphore = image_available_semaphores_[current_frame_]->Handle();
  VkSemaphore render_finished_semaphore = render_finished_semaphores_[current_frame_]->Handle();

  VkFence fence = in_flight_fences_[current_frame_]->Handle();
  VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  if (window_) {
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &image_available_semaphore;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &render_finished_semaphore;
  }

  if (vkQueueSubmit(graphics_queue_->Handle(), 1, &submit_info, fence) != VK_SUCCESS) {
    throw std::runtime_error("Failed to submit command buffer");
  }

  if (window_) {
    VkSwapchainKHR swapchain = swapchain_->Handle();
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_finished_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain;
    present_info.pImageIndices = &image_index_;

    VkResult result = vkQueuePresentKHR(present_queue_->Handle(), &present_info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      device_->WaitIdle();
      swapchain_.reset();
      device_->CreateSwapchain(surface_.get(), &swapchain_);
    } else if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to present image");
    }
  }

  current_frame_ = (current_frame_ + 1) % max_frames_in_flight_;
}
