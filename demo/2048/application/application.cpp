#include "application.h"

#include <algorithm>
#include <map>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "listener.h"
#include "model.h"
#include "stb_image_write.h"

namespace {

#include "built_in_shaders.inl"

// GLFW has no user data slot left for cursor-enter events: the graphics window
// owns the window user pointer.
Application *cursor_enter_application = nullptr;

struct ResolveParams {
  float second_frame_alpha;
  uint32_t scale;
  uint32_t padding[2];
};

int ChooseSupersampleScale(glm::ivec2 size) {
  const int max_extent = std::max(size.x, size.y);
  if (max_extent <= 2048) {
    return 3;
  }
  if (max_extent <= 4096) {
    return 2;
  }
  return 1;
}

}  // namespace

Application::Application(const std::string &name, int width, int height, graphics::BackendAPI api) : name_(name) {
  if (!graphics::SupportBackendAPI(api) || graphics::CreateCore(api, graphics::Core::Settings{}, &core_) != 0 ||
      !core_) {
    throw std::runtime_error("Requested graphics backend is unavailable");
  }
  if (core_->InitializeLogicalDeviceAutoSelect(false) != 0) {
    throw std::runtime_error("No compatible graphics device found");
  }
  LogInfo("Backend API: {}", graphics::BackendAPIString(core_->API()));
  LogInfo("Device Name: {}", core_->DeviceName());

  core_->CreateWindowObject(width, height, name_, false, true, &window_);

  mouse_move_callback_ = window_->MouseMoveEvent().RegisterCallback(
      [this](double xpos, double ypos) { NotifyListeners(&Listener::OnCursorPos, xpos, ypos); });
  mouse_button_callback_ =
      window_->MouseButtonEvent().RegisterCallback([this](int button, int action, int mods, double, double) {
        NotifyListeners(&Listener::OnMouseButton, button, action, mods);
      });

  cursor_enter_application = this;
  glfwSetCursorEnterCallback(window_->GLFWWindow(), [](GLFWwindow *, int entered) {
    if (cursor_enter_application) {
      cursor_enter_application->NotifyListeners(&Listener::OnCursorEnter, entered);
    }
  });
}

Application::~Application() {
  if (cursor_enter_application == this) {
    cursor_enter_application = nullptr;
  }
}

void Application::Run(int max_frames) {
  OnInit();
  int frames = 0;
  while (!window_->ShouldClose() && (max_frames <= 0 || frames < max_frames)) {
    glfwPollEvents();
    OnUpdate();
    OnRender();
    frames++;
  }
  if (!screenshot_path_.empty()) {
    SaveScreenshot();
  }
  OnClose();
}

void Application::DrawModel(DeviceModel *device_model, const InstanceInfo &instance_info) {
  main_frame_.instances.emplace_back(device_model, instance_info);
}

void Application::CaptureSecondFrame(float alpha) {
  second_frame_.instances = std::move(main_frame_.instances);
  main_frame_.instances.clear();
  second_frame_alpha_ = alpha;
}

void Application::RegisterListener(Listener *listener) {
  listeners_.insert(listener);
}

void Application::UnregisterListener(Listener *listener) {
  listeners_.erase(listener);
}

void Application::CustomOnUpdate() {
}

void Application::CustomOnClose() {
}

void Application::CustomOnInit() {
}

void Application::OnFramebufferResize() {
}

void Application::OnInit() {
  const auto shader_code = GetShaderCode("shaders/super.hlsl");
  core_->CreateShader(shader_code, "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(shader_code, "PSMain", "ps_6_0", &pixel_shader_);
  core_->CreateProgram({graphics::IMAGE_FORMAT_R8G8B8A8_UNORM}, graphics::IMAGE_FORMAT_D32_SFLOAT, &program_);
  program_->BindShader(vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(pixel_shader_.get(), graphics::SHADER_TYPE_PIXEL);
  program_->AddInputBinding(sizeof(Vertex));
  program_->AddInputBinding(sizeof(uint32_t), true);
  program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT2, offsetof(Vertex, position));
  program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT4, offsetof(Vertex, color));
  program_->AddInputAttribute(1, graphics::INPUT_TYPE_UINT, 0);
  program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program_->SetCullMode(graphics::CULL_MODE_NONE);
  program_->Finalize();

  const auto resolve_code = GetShaderCode("shaders/resolve.hlsl");
  core_->CreateShader(resolve_code, "VSMain", "vs_6_0", &resolve_vertex_shader_);
  core_->CreateShader(resolve_code, "PSMain", "ps_6_0", &resolve_pixel_shader_);
  core_->CreateProgram({graphics::IMAGE_FORMAT_R8G8B8A8_UNORM}, graphics::IMAGE_FORMAT_UNDEFINED, &resolve_program_);
  resolve_program_->BindShader(resolve_vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
  resolve_program_->BindShader(resolve_pixel_shader_.get(), graphics::SHADER_TYPE_PIXEL);
  resolve_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  resolve_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  resolve_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  resolve_program_->SetCullMode(graphics::CULL_MODE_NONE);
  resolve_program_->Finalize();

  core_->CreateBuffer(sizeof(glm::mat4), graphics::BUFFER_TYPE_DYNAMIC, &global_uniform_buffer_);
  core_->CreateBuffer(sizeof(ResolveParams), graphics::BUFFER_TYPE_DYNAMIC, &resolve_uniform_buffer_);
  const uint32_t first_instance_index = 0;
  core_->CreateBuffer(sizeof(uint32_t), graphics::BUFFER_TYPE_DYNAMIC, &instance_index_buffer_);
  instance_index_buffer_->UploadData(&first_instance_index, sizeof(uint32_t));
  core_->CreateBuffer(sizeof(InstanceInfo), graphics::BUFFER_TYPE_DYNAMIC, &main_frame_.instance_buffer);
  core_->CreateBuffer(sizeof(InstanceInfo), graphics::BUFFER_TYPE_DYNAMIC, &second_frame_.instance_buffer);

  BuildScreenFrameObjects();

  fps_start_time_ = glfwGetTime();
  CustomOnInit();
}

void Application::OnUpdate() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window_->GLFWWindow(), &width, &height);
  if (width > 0 && height > 0 && glm::ivec2{width, height} != framebuffer_size_) {
    BuildScreenFrameObjects();
    OnFramebufferResize();
  }

  main_frame_.instances.clear();
  second_frame_.instances.clear();
  second_frame_alpha_ = 0.0f;

  CustomOnUpdate();
  UpdateTitle();
}

void Application::OnRender() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window_->GLFWWindow(), &width, &height);
  if (width <= 0 || height <= 0) {
    // Minimized windows have no framebuffer to present.
    return;
  }

  // Instance indices are fed through a per-instance vertex stream so that the
  // first instance offset works identically on every backend.
  const size_t max_instances = std::max(main_frame_.instances.size(), second_frame_.instances.size());
  if (max_instances * sizeof(uint32_t) > instance_index_buffer_->Size()) {
    std::vector<uint32_t> indices(std::max<size_t>(max_instances, instance_index_buffer_->Size() / 2));
    for (size_t i = 0; i < indices.size(); i++) {
      indices[i] = static_cast<uint32_t>(i);
    }
    instance_index_buffer_->Resize(indices.size() * sizeof(uint32_t));
    instance_index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));
  }

  std::unique_ptr<graphics::CommandContext> context;
  core_->CreateCommandContext(&context);

  const bool use_second_frame = second_frame_alpha_ > 0.0f;
  if (use_second_frame) {
    RenderFrameTarget(context.get(), second_frame_);
  }
  RenderFrameTarget(context.get(), main_frame_);

  ResolveParams params{use_second_frame ? second_frame_alpha_ : 0.0f, static_cast<uint32_t>(supersample_scale_)};
  resolve_uniform_buffer_->UploadData(&params, sizeof(params));

  auto *second_image = use_second_frame ? second_frame_.color_image.get() : main_frame_.color_image.get();
  context->CmdBeginRendering({present_image_.get()}, nullptr);
  context->CmdBindProgram(resolve_program_.get());
  context->CmdBindResources(0, {resolve_uniform_buffer_.get()});
  context->CmdBindResources(1, {main_frame_.color_image.get()});
  context->CmdBindResources(2, {second_image});
  context->CmdSetViewport({0.0f, 0.0f, float(framebuffer_size_.x), float(framebuffer_size_.y), 0.0f, 1.0f});
  context->CmdSetScissor({{0, 0}, {uint32_t(framebuffer_size_.x), uint32_t(framebuffer_size_.y)}});
  context->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  context->CmdDraw(3, 1, 0, 0);
  context->CmdEndRendering();

  context->CmdPresent(window_.get(), present_image_.get());
  core_->SubmitCommandContext(context.get());
}

void Application::RenderFrameTarget(graphics::CommandContext *context, FrameTarget &target) {
  auto &instances = target.instances;
  // Group instances of the same model into a single draw call. The stable sort
  // keeps submission order within a model; the depth buffer resolves layering.
  std::stable_sort(instances.begin(), instances.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

  if (!instances.empty()) {
    std::vector<InstanceInfo> instance_infos;
    instance_infos.reserve(instances.size());
    for (auto &instance : instances) {
      instance_infos.push_back(instance.second);
    }
    const size_t data_size = instance_infos.size() * sizeof(InstanceInfo);
    if (data_size > target.instance_buffer->Size()) {
      target.instance_buffer->Resize(data_size);
    }
    target.instance_buffer->UploadData(instance_infos.data(), data_size);
  }

  const auto extent = target.color_image->Extent();
  context->CmdClearImage(target.color_image.get(), {{clear_color_.r, clear_color_.g, clear_color_.b, clear_color_.a}});
  context->CmdClearImage(depth_image_.get(), {{1.0f}});
  context->CmdBeginRendering({target.color_image.get()}, depth_image_.get());
  context->CmdBindProgram(program_.get());
  context->CmdBindResources(0, {target.instance_buffer.get()});
  context->CmdBindResources(1, {global_uniform_buffer_.get()});
  context->CmdBindVertexBuffers(1, {instance_index_buffer_.get()}, {0});
  context->CmdSetViewport({0.0f, 0.0f, float(extent.width), float(extent.height), 0.0f, 1.0f});
  context->CmdSetScissor({{0, 0}, extent});
  context->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

  for (size_t head = 0, tail = 0; head < instances.size(); head = tail) {
    while (tail < instances.size() && instances[head].first == instances[tail].first) {
      tail++;
    }
    auto *device_model = instances[head].first;
    if (!device_model->IndexCount()) {
      continue;
    }
    context->CmdBindVertexBuffers(0, {device_model->VertexBuffer()}, {0});
    context->CmdBindIndexBuffer(device_model->IndexBuffer(), 0);
    context->CmdDrawIndexed(device_model->IndexCount(), uint32_t(tail - head), 0, 0, uint32_t(head));
  }
  context->CmdEndRendering();
}

void Application::OnClose() {
  core_->WaitGPU();

  CustomOnClose();

  main_frame_ = {};
  second_frame_ = {};
  depth_image_.reset();
  present_image_.reset();
  instance_index_buffer_.reset();
  resolve_uniform_buffer_.reset();
  global_uniform_buffer_.reset();
  resolve_program_.reset();
  resolve_pixel_shader_.reset();
  resolve_vertex_shader_.reset();
  program_.reset();
  pixel_shader_.reset();
  vertex_shader_.reset();

  glfwSetCursorEnterCallback(window_->GLFWWindow(), nullptr);
  window_->MouseMoveEvent().UnregisterCallback(mouse_move_callback_);
  window_->MouseButtonEvent().UnregisterCallback(mouse_button_callback_);
}

void Application::BuildScreenFrameObjects() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window_->GLFWWindow(), &width, &height);
  framebuffer_size_ = {std::max(width, 1), std::max(height, 1)};
  supersample_scale_ = ChooseSupersampleScale(framebuffer_size_);
  const auto sample_size = framebuffer_size_ * supersample_scale_;

  core_->WaitGPU();
  main_frame_.color_image.reset();
  second_frame_.color_image.reset();
  depth_image_.reset();
  present_image_.reset();
  core_->CreateImage(sample_size.x, sample_size.y, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &main_frame_.color_image);
  core_->CreateImage(sample_size.x, sample_size.y, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &second_frame_.color_image);
  core_->CreateImage(sample_size.x, sample_size.y, graphics::IMAGE_FORMAT_D32_SFLOAT, &depth_image_);
  core_->CreateImage(framebuffer_size_.x, framebuffer_size_.y, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &present_image_);

  // Maps framebuffer pixel coordinates (y pointing down) to normalized device coordinates.
  glm::mat4 view{1.0f};
  view[0][0] = 2.0f / float(framebuffer_size_.x);
  view[1][1] = -2.0f / float(framebuffer_size_.y);
  view[3][0] = -1.0f;
  view[3][1] = 1.0f;
  global_uniform_buffer_->UploadData(&view, sizeof(view));
}

void Application::UpdateTitle() {
  fps_frames_++;
  const double now = glfwGetTime();
  if (now - fps_start_time_ >= 1.0) {
    window_->SetTitle(fmt::format("{} [{}] FPS: {:.1f}", name_, graphics::BackendAPIString(core_->API()),
                                  fps_frames_ / (now - fps_start_time_)));
    fps_frames_ = 0;
    fps_start_time_ = now;
  }
}

void Application::SaveScreenshot() {
  core_->WaitGPU();
  const auto extent = present_image_->Extent();
  std::vector<uint8_t> pixels(size_t(extent.width) * extent.height * 4);
  present_image_->DownloadData(pixels.data());
  if (stbi_write_png(screenshot_path_.c_str(), int(extent.width), int(extent.height), 4, pixels.data(),
                     int(extent.width) * 4)) {
    LogInfo("Screenshot saved to {}", screenshot_path_);
  } else {
    LogError("Failed to save screenshot to {}", screenshot_path_);
  }
}
