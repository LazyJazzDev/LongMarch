#include "application.h"

#include <algorithm>
#include <cstring>
#include <map>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "listener.h"
#include "model.h"
#include "stb_image_write.h"

namespace {

#include "built_in_shaders.inl"

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

Application::Application(const std::string &name, int width, int height, rhi::BackendAPI api)
    : name_(name),
      api_(api),
      initial_width_(width),
      initial_height_(height) {
}

Application::~Application() {
  upload_context_.Release();
  WaitFrames();
  if (!window_)
    return;
  window_->CursorEnterEvent().UnregisterCallback(cursor_enter_callback_);
  window_->FocusEvent().UnregisterCallback(focus_callback_);
  window_->MouseMoveEvent().UnregisterCallback(mouse_move_callback_);
  window_->MouseButtonEvent().UnregisterCallback(mouse_button_callback_);
}

bool Application::Run(int max_frames) {
  if (!OnInit()) {
    OnClose();
    return false;
  }
  int frames = 0;
  while (!failed_ && !window_->ShouldClose() && (max_frames <= 0 || frames < max_frames)) {
    // Bound CPU/GPU overlap without making every upload a synchronous transfer.
    auto &previous = submissions_[submission_index_];
    if (previous && !Check(previous->WaitHost(), "Wait for frame slot"))
      break;
    previous.Release();
    if (!Check(rhi::PollEvents(), "Poll window events"))
      break;
    OnUpdate();
    if (failed_ || !OnRender())
      break;
    frames++;
  }
  WaitFrames();
  if (!failed_ && !screenshot_path_.empty())
    SaveScreenshot();
  OnClose();
  return !failed_;
}

void Application::DrawModel(DeviceModel *device_model, const InstanceInfo &instance_info) {
  main_frame_.instances.emplace_back(device_model, instance_info);
}

void Application::DrawModelWithData(DeviceModel *device_model,
                                    InstanceInfo instance_info,
                                    const std::vector<uint32_t> &data) {
  instance_info.extra.y = uint32_t(main_frame_.data.size() * sizeof(uint32_t));
  main_frame_.data_instances.push_back(main_frame_.instances.size());
  main_frame_.data.insert(main_frame_.data.end(), data.begin(), data.end());
  main_frame_.instances.emplace_back(device_model, instance_info);
}

void Application::CaptureSecondFrame(float alpha) {
  second_frame_.instances = std::move(main_frame_.instances);
  second_frame_.data = std::move(main_frame_.data);
  second_frame_.data_instances = std::move(main_frame_.data_instances);
  main_frame_.instances.clear();
  main_frame_.data.clear();
  main_frame_.data_instances.clear();
  second_frame_alpha_ = alpha;
}

void Application::RegisterListener(Listener *listener) {
  if (listeners_.insert(listener).second)
    ++listener_revision_;
}

void Application::UnregisterListener(Listener *listener) {
  if (listeners_.erase(listener))
    ++listener_revision_;
}

void Application::CustomOnUpdate() {
}

void Application::CustomOnClose() {
}

void Application::CustomOnInit() {
}

void Application::OnFramebufferResize() {
}

bool Application::OnInit() {
  core_ = rhi::CreateCore(api_);
  if (!Check(bool(core_), "Create RHI core"))
    return false;
  device_ = core_->CreateDeviceByAutoSelection();
  if (!Check(bool(device_), "Select RHI device"))
    return false;
  queue_ = device_->GetQueue(rhi::QUEUE_TYPE_MAIN);
  if (!Check(bool(queue_), "Get main queue"))
    return false;
  auto *window = device_->CreateGLFWWindow({initial_width_, initial_height_,
                                            fmt::format("[{}] {} FPS: 0.0", rhi::BackendAPIName(api_), name_),
                                            rhi::PRESENT_MODE_CONTINUOUS});
  if (!Check(window != nullptr, "Create RHI window"))
    return false;
  window_ = window;
  LogInfo("Backend API: {}", rhi::BackendAPIName(api_));
  LogInfo("Device Name: {}", device_->Name());
  mouse_move_callback_ = window_->MouseMoveEvent().RegisterCallback(
      [this](double x, double y) { NotifyListeners(&Listener::OnCursorPos, x, y); });
  mouse_button_callback_ =
      window_->MouseButtonEvent().RegisterCallback([this](int button, int action, int mods, double, double) {
        NotifyListeners(&Listener::OnMouseButton, button, action, mods);
      });
  cursor_enter_callback_ = window_->CursorEnterEvent().RegisterCallback(
      [this](bool entered) { NotifyListeners(&Listener::OnCursorEnter, int(entered)); });
  focus_callback_ =
      window_->FocusEvent().RegisterCallback([this](bool focused) { NotifyListeners(&Listener::OnFocus, focused); });

  vertex_shader_ =
      device_->CreateShaderFromSource(GetShaderCode("shaders/super.slang"), "VSMain", rhi::SHADER_STAGE_VERTEX);
  pixel_shader_ =
      device_->CreateShaderFromSource(GetShaderCode("shaders/super.slang"), "PSMain", rhi::SHADER_STAGE_FRAGMENT);
  resolve_vertex_shader_ =
      device_->CreateShaderFromSource(GetShaderCode("shaders/resolve.slang"), "VSMain", rhi::SHADER_STAGE_VERTEX);
  resolve_pixel_shader_ =
      device_->CreateShaderFromSource(GetShaderCode("shaders/resolve.slang"), "PSMain", rhi::SHADER_STAGE_FRAGMENT);
  if (!Check(vertex_shader_ && pixel_shader_ && resolve_vertex_shader_ && resolve_pixel_shader_,
             "Compile demo shaders"))
    return false;

  const rhi::BindingLayoutDesc instances{{{0, rhi::BINDING_BYTE_ADDRESS_BUFFER}}};
  const rhi::BindingLayoutDesc uniform{{{0, rhi::BINDING_UNIFORM_BUFFER}}};
  const rhi::BindingLayoutDesc image{{{0, rhi::BINDING_TEXTURE2D}}};
  rhi::ProgramDesc desc;
  desc.vertex_shader = vertex_shader_.Get();
  desc.fragment_shader = pixel_shader_.Get();
  desc.vertex_bindings = {{sizeof(Vertex)}, {sizeof(uint32_t), true}};
  desc.vertex_attributes = {{0, 0, rhi::INPUT_TYPE_FLOAT2, offsetof(Vertex, position)},
                            {1, 0, rhi::INPUT_TYPE_FLOAT4, offsetof(Vertex, color)},
                            {2, 1, rhi::INPUT_TYPE_UINT, 0}};
  desc.depth_stencil = {rhi::IMAGE_FORMAT_D32_SFLOAT, true, true};
  desc.rasterizer.cull_mode = rhi::CULL_MODE_NONE;
  desc.pipeline_layout.groups = {instances, uniform};
  program_ = device_->CreateProgram(desc);
  rhi::ProgramDesc resolve;
  resolve.vertex_shader = resolve_vertex_shader_.Get();
  resolve.fragment_shader = resolve_pixel_shader_.Get();
  resolve.rasterizer.cull_mode = rhi::CULL_MODE_NONE;
  resolve.pipeline_layout.groups = {uniform, image, image};
  resolve_program_ = device_->CreateProgram(resolve);
  instance_group_ = device_->CreateBindingGroup(instances);
  global_group_ = device_->CreateBindingGroup(uniform);
  resolve_group_ = device_->CreateBindingGroup(uniform);
  main_image_group_ = device_->CreateBindingGroup(image);
  second_image_group_ = device_->CreateBindingGroup(image);
  if (!Check(program_ && resolve_program_ && instance_group_ && global_group_ && resolve_group_ && main_image_group_ &&
                 second_image_group_,
             "Create render pipelines and bindings"))
    return false;

  // D3D12 constant-buffer views require a 256-byte backing allocation.
  global_uniform_buffer_ = CreateBuffer(256);
  resolve_uniform_buffer_ = CreateBuffer(256);
  main_frame_.instance_buffer = CreateBuffer(sizeof(InstanceInfo));
  second_frame_.instance_buffer = CreateBuffer(sizeof(InstanceInfo));
  const uint32_t first_index = 0;
  if (failed_ || !UploadBuffer(instance_index_buffer_, &first_index, sizeof(first_index)))
    return false;
  instance_index_count_ = 1;
  if (!BuildScreenFrameObjects())
    return false;
  title_time_ = grassland::GetTimeSeconds();
  custom_initialized_ = true;
  CustomOnInit();
  return !failed_;
}

void Application::OnUpdate() {
  const auto size = GetFramebufferSize();
  const int width = size.x, height = size.y;
  if (width > 0 && height > 0 && glm::ivec2{width, height} != framebuffer_size_) {
    if (!BuildScreenFrameObjects())
      return;
    OnFramebufferResize();
  }

  for (auto *target : {&main_frame_, &second_frame_}) {
    target->instances.clear();
    target->data.clear();
    target->data_instances.clear();
  }
  second_frame_alpha_ = 0.0f;

  CustomOnUpdate();
  UpdateTitle();
}

bool Application::OnRender() {
  const auto size = GetFramebufferSize();
  if (size.x <= 0 || size.y <= 0)
    return true;
  auto *window = window_.Get();
  if (window->IsSwapChainInvalid()) {
    // No swapchain images are held by the demo; drain before rebuilding.
    if (!WaitFrames() || !Check(window->RebuildSwapChain(), "Rebuild swapchain"))
      return false;
    if (window->IsSwapChainInvalid())
      return true;
  }

  const size_t max_instances = std::max(main_frame_.instances.size(), second_frame_.instances.size());
  if (max_instances > instance_index_count_) {
    std::vector<uint32_t> indices(std::max(max_instances, instance_index_count_ * 2));
    for (size_t i = 0; i < indices.size(); i++)
      indices[i] = static_cast<uint32_t>(i);
    if (!UploadBuffer(instance_index_buffer_, indices.data(), indices.size() * sizeof(uint32_t)))
      return false;
    instance_index_count_ = indices.size();
  }
  if (!upload_context_)
    upload_context_ = queue_->CreateCommandContext();
  if (!Check(bool(upload_context_), "Create frame commands"))
    return false;
  auto *context = upload_context_.Get();
  const bool use_second_frame = second_frame_alpha_ > 0.0f;
  if (use_second_frame && !RenderFrameTarget(context, second_frame_))
    return false;
  if (!RenderFrameTarget(context, main_frame_))
    return false;
  ResolveParams params{use_second_frame ? second_frame_alpha_ : 0.0f, static_cast<uint32_t>(supersample_scale_)};
  if (!UploadBuffer(resolve_uniform_buffer_, &params, sizeof(params)))
    return false;
  auto *second_image = use_second_frame ? second_frame_.color_image.Get() : main_frame_.color_image.Get();
  if (!Check(resolve_group_->BindResource(0, resolve_uniform_buffer_.Get()) &&
                 main_image_group_->BindResource(0, main_frame_.color_image.Get()) &&
                 second_image_group_->BindResource(0, second_image),
             "Bind resolve resources"))
    return false;
  rhi::RenderingInfo rendering;
  rendering.color_attachments = {{present_image_.Get(), rhi::ATTACHMENT_LOAD_OP_DISCARD}};
  if (!Check(context->CmdBeginRendering(rendering) && context->CmdBindProgram(resolve_program_.Get()) &&
                 context->CmdBindGroup(0, resolve_group_.Get()) && context->CmdBindGroup(1, main_image_group_.Get()) &&
                 context->CmdBindGroup(2, second_image_group_.Get()) &&
                 context->CmdSetViewport({0, 0, float(framebuffer_size_.x), float(framebuffer_size_.y), 0, 1}) &&
                 context->CmdSetScissor({0, 0, uint32_t(framebuffer_size_.x), uint32_t(framebuffer_size_.y)}) &&
                 context->CmdDraw(3) && context->CmdEndRendering() &&
                 context->CmdPresentImage(window, present_image_.Get(), rhi::COLOR_SPACE_SRGB_NONLINEAR),
             "Record resolve and presentation"))
    return false;
  submissions_[submission_index_] = context->Submit();
  upload_context_.Release();
  if (!Check(bool(submissions_[submission_index_]), "Submit frame"))
    return false;
  submission_index_ = (submission_index_ + 1) % submissions_.size();
  return true;
}

bool Application::RenderFrameTarget(rhi::ICommandContext *context, FrameTarget &target) {
  auto &instances = target.instances;

  // Group instances of the same model into a single draw call without sorting
  // or moving the large InstanceInfo records: sort the short list of runs of
  // consecutive instances instead. The stable sort keeps submission order within
  // a model; the depth buffer resolves layering.
  struct Batch {
    DeviceModel *model;
    size_t head, count;
  };

  std::vector<Batch> batches;
  for (size_t head = 0, tail = 0; head < instances.size(); head = tail) {
    while (tail < instances.size() && instances[head].first == instances[tail].first)
      ++tail;
    batches.push_back({instances[head].first, head, tail - head});
  }
  std::stable_sort(batches.begin(), batches.end(),
                   [](const auto &a, const auto &b) { return std::less<DeviceModel *>{}(a.model, b.model); });

  if (!instances.empty()) {
    // One upload: instance records, then the data words they reference.
    const size_t records_size = instances.size() * sizeof(InstanceInfo);
    const size_t data_size = target.data.size() * sizeof(uint32_t);
    auto &upload = target.upload;
    upload.resize(records_size + data_size);
    auto *records = reinterpret_cast<InstanceInfo *>(upload.data());
    for (size_t i = 0; i < instances.size(); i++) {
      records[i] = instances[i].second;
    }
    for (size_t index : target.data_instances) {
      records[index].extra.y += uint32_t(records_size);
    }
    if (data_size)
      std::memcpy(upload.data() + records_size, target.data.data(), data_size);
    if (!UploadBuffer(target.instance_buffer, upload.data(), upload.size()))
      return false;
  }

  const auto extent = target.color_image->GetDesc();
  if (!Check(instance_group_->BindResource(0, target.instance_buffer.Get()) &&
                 global_group_->BindResource(0, global_uniform_buffer_.Get()),
             "Bind scene resources"))
    return false;
  rhi::RenderingInfo rendering;
  rendering.color_attachments = {{target.color_image.Get(),
                                  rhi::ATTACHMENT_LOAD_OP_CLEAR,
                                  rhi::ATTACHMENT_STORE_OP_STORE,
                                  {clear_color_.r, clear_color_.g, clear_color_.b, clear_color_.a}}};
  rendering.depth_stencil.image = depth_image_.Get();
  rendering.depth_stencil.depth_load_op = rhi::ATTACHMENT_LOAD_OP_CLEAR;
  if (!Check(context->CmdBeginRendering(rendering) && context->CmdBindProgram(program_.Get()) &&
                 context->CmdBindGroup(0, instance_group_.Get()) && context->CmdBindGroup(1, global_group_.Get()) &&
                 context->CmdBindVertexBuffer(1, instance_index_buffer_.Get()) &&
                 context->CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1}) &&
                 context->CmdSetScissor({0, 0, extent.width, extent.height}),
             "Begin scene rendering"))
    return false;

  for (const auto &batch : batches) {
    auto *device_model = batch.model;
    if (!device_model->IndexCount())
      continue;
    if (!Check(
            context->CmdBindVertexBuffer(0, device_model->VertexBuffer()) &&
                context->CmdBindIndexBuffer(device_model->IndexBuffer()) &&
                context->CmdDrawIndexed(device_model->IndexCount(), uint32_t(batch.count), 0, 0, uint32_t(batch.head)),
            "Draw model batch"))
      return false;
  }
  return Check(context->CmdEndRendering(), "End scene rendering");
}

void Application::OnClose() {
  // Discard recordings before releasing resources, including on an error path.
  upload_context_.Release();
  WaitFrames();
  if (custom_initialized_) {
    CustomOnClose();
    custom_initialized_ = false;
  }
  main_frame_ = {};
  second_frame_ = {};
  instance_group_.Release();
  global_group_.Release();
  resolve_group_.Release();
  main_image_group_.Release();
  second_image_group_.Release();
  depth_image_.Release();
  present_image_.Release();
  instance_index_buffer_.Release();
  resolve_uniform_buffer_.Release();
  global_uniform_buffer_.Release();
  resolve_program_.Release();
  resolve_pixel_shader_.Release();
  resolve_vertex_shader_.Release();
  program_.Release();
  pixel_shader_.Release();
  vertex_shader_.Release();
}

bool Application::BuildScreenFrameObjects() {
  if (!WaitFrames())
    return false;
  const auto size = GetFramebufferSize();
  framebuffer_size_ = glm::max(size, glm::ivec2{1});
  supersample_scale_ = ChooseSupersampleScale(framebuffer_size_);
  const auto sample_size = framebuffer_size_ * supersample_scale_;
  const rhi::ImageDesc color{uint32_t(sample_size.x), uint32_t(sample_size.y), rhi::IMAGE_FORMAT_R8G8B8A8_UNORM};
  main_frame_.color_image = device_->CreateImage(color);
  second_frame_.color_image = device_->CreateImage(color);
  depth_image_ = device_->CreateImage({color.width, color.height, rhi::IMAGE_FORMAT_D32_SFLOAT});
  present_image_ = device_->CreateImage(
      {uint32_t(framebuffer_size_.x), uint32_t(framebuffer_size_.y), rhi::IMAGE_FORMAT_R8G8B8A8_UNORM});
  if (!Check(main_frame_.color_image && second_frame_.color_image && depth_image_ && present_image_,
             "Create frame targets"))
    return false;
  glm::mat4 view{1.0f};
  view[0][0] = 2.0f / float(framebuffer_size_.x);
  view[1][1] = -2.0f / float(framebuffer_size_.y);
  view[3][0] = -1.0f;
  view[3][1] = 1.0f;
  return UploadBuffer(global_uniform_buffer_, &view, sizeof(view));
}

void Application::UpdateTitle() {
  // FPSCounter averages over the frames of the last second; the title shows it
  // four times a second.
  fps_counter_.TickFrame();
  const double now = grassland::GetTimeSeconds();
  if (now - title_time_ >= 0.25) {
    window_->SetTitle(fmt::format("[{}] {} FPS: {:.1f}", rhi::BackendAPIName(api_), name_, fps_counter_.GetFPS()));
    title_time_ = now;
  }
}

bool Application::SaveScreenshot() {
  const auto desc = present_image_->GetDesc();
  rhi::ObjPtr<rhi::ICommandContext> context = queue_->CreateCommandContext();
  if (!Check(bool(context), "Create screenshot commands"))
    return false;
  rhi::ObjPtr<rhi::IDownloadToken> download = context->CmdDownloadImage(present_image_.Get());
  if (!Check(bool(download), "Record screenshot download"))
    return false;
  rhi::ObjPtr<rhi::ISubmissionToken> submission = context->Submit();
  if (!Check(submission && submission->WaitHost() && download->WaitHost(), "Download screenshot"))
    return false;
  std::vector<uint8_t> pixels(size_t(desc.width) * desc.height * 4);
  if (!Check(download->CopyTo(pixels.data(), pixels.size()), "Copy screenshot pixels"))
    return false;
  if (!Check(stbi_write_png(screenshot_path_.c_str(), int(desc.width), int(desc.height), 4, pixels.data(),
                            int(desc.width) * 4) != 0,
             "Write screenshot PNG"))
    return false;
  LogInfo("Screenshot saved to {}", screenshot_path_);
  return true;
}

bool Application::Check(bool success, const char *operation) {
  if (!success) {
    if (!failed_)
      LogError("{} failed", operation);
    failed_ = true;
  }
  return success;
}

bool Application::WaitFrames() {
  bool success = true;
  for (auto &submission : submissions_) {
    if (submission)
      success = Check(submission->WaitHost(), "Wait for GPU work") && success;
    submission.Release();
  }
  return success;
}

rhi::ObjPtr<rhi::IBuffer> Application::CreateBuffer(uint64_t size) {
  if (failed_ || !device_)
    return {};
  rhi::ObjPtr<rhi::IBuffer> buffer = device_->CreateBuffer({std::max<uint64_t>(size, 4)});
  Check(bool(buffer), "Allocate buffer");
  return buffer;
}

bool Application::UploadBuffer(rhi::ObjPtr<rhi::IBuffer> &buffer, const void *data, uint64_t size) {
  if (failed_)
    return false;
  if (!buffer || buffer->GetDesc().size < size) {
    auto replacement = CreateBuffer(std::max(size, buffer ? buffer->GetDesc().size * 2 : uint64_t{4}));
    if (!replacement)
      return false;
    buffer = std::move(replacement);
  }
  if (!size)
    return true;
  if (!upload_context_)
    upload_context_ = queue_->CreateCommandContext();
  return Check(upload_context_ && upload_context_->CmdUploadBuffer(buffer.Get(), 0, data, size), "Upload buffer");
}

glm::ivec2 Application::GetWindowSize() const {
  glm::ivec2 size{};
  glfwGetWindowSize(window_->GLFWWindow(), &size.x, &size.y);
  return size;
}

glm::ivec2 Application::GetFramebufferSize() const {
  glm::ivec2 size{};
  glfwGetFramebufferSize(window_->GLFWWindow(), &size.x, &size.y);
  return size;
}

glm::dvec2 Application::GetCursorPosition() const {
  glm::dvec2 position{};
  glfwGetCursorPos(window_->GLFWWindow(), &position.x, &position.y);
  return position;
}
