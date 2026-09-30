#include "app.h"

#include <glm/gtc/matrix_transform.hpp>

Application::Application(grassland::graphics::BackendAPI api) {
  grassland::graphics::CreateCore(api, grassland::graphics::Core::Settings{}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);

  grassland::LogInfo("Device Name: {}", core_->DeviceName());
  grassland::LogInfo("- Ray Tracing Support: {}", core_->DeviceRayTracingSupport());
}

Application::~Application() {
  core_.reset();
}

void Application::OnInit() {
  alive_ = true;
  core_->CreateWindowObject(1280, 720, std::string("Snowberg GUI Demo"), false, true, &window_);
  ui_ = std::make_unique<snowberg::gui::Context>(core_.get(), window_.get(), snowberg::gui::DefaultFont());
  world_panel_ =
      std::make_unique<snowberg::gui::WorldPanel>(core_.get(), window_.get(), 320, 240, snowberg::gui::DefaultFont());
  auto transform = glm::translate(glm::mat4(1), glm::vec3{0.2f, 0.7f, 0.0f});
  transform = glm::rotate(transform, glm::radians(-20.0f), glm::vec3{0, 1, 0});
  world_panel_->SetTransform(glm::scale(transform, glm::vec3{2.0f, -1.5f, 1.0f}));
  core_->CreateImage(window_->GetFramebufferSize().x, window_->GetFramebufferSize().y,
                     grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &frame_image_);
}

void Application::OnClose() {
  core_->WaitGPU();
  world_panel_.reset();
  depth_image_.reset();
  ui_.reset();
  frame_image_.reset();
  window_.reset();
}

void Application::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }

  if (alive_) {
    ui_->BeginFrame();
    ui_->BeginPanel("demo", 32, 32, 340, "Snowberg GUI");
    ui_->Text("A unified interface on grassland/graphics");
    ui_->Separator();
    ui_->Checkbox("Animate", &animated_);
    ui_->Slider("Intensity", &intensity_, 0.0f, 1.0f);
    ui_->Checkbox("Particles", &particles_);
    if (particles_) {
      const float time = animated_ ? static_cast<float>(grassland::GetTimeSeconds()) : 0.0f;
      ui_->Custom("particles", 60, [time](snowberg::gui::Canvas &canvas) {
        canvas.ParticleField(32, time, {0.04f, 0.43f, 0.91f, 0.38f});
      });
    }
    if (ui_->Button("Reset"))
      intensity_ = 0.5f;
    ui_->EndPanel();
    const auto size = glm::max(window_->GetSize(), glm::ivec2{1});
    view_projection_ = glm::perspectiveZO(glm::radians(45.0f), float(size.x) / size.y, 0.1f, 20.0f) *
                       glm::lookAt(glm::vec3{0, 0, 5}, glm::vec3{0}, glm::vec3{0, 1, 0});
    const auto cursor = window_->GetCursorPosition();
    const glm::vec2 ndc{float(2.0 * cursor.x / size.x - 1.0), float(1.0 - 2.0 * cursor.y / size.y)};
    const auto inverse = glm::inverse(view_projection_);
    auto near_point = inverse * glm::vec4{ndc, 0, 1};
    auto far_point = inverse * glm::vec4{ndc, 1, 1};
    near_point /= near_point.w;
    far_point /= far_point.w;
    const auto hit = world_panel_->RayHit(glm::vec3(near_point), glm::normalize(glm::vec3(far_point - near_point)));
    world_panel_->BeginFrame(ui_->WantsPointer() ? std::nullopt : hit,
                             window_->IsMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT));
    auto &controls = world_panel_->Controls();
    controls.BeginPanel("world", 0, 0, 320, "World controls");
    controls.Checkbox("Animate", &animated_);
    controls.Slider("Intensity", &intensity_, 0.0f, 1.0f);
    controls.EndPanel();
  }
}

void Application::OnRender() {
  const auto size = window_->GetFramebufferSize();
  if (size.x <= 0 || size.y <= 0)
    return;
  if (!frame_image_ || frame_image_->Extent().width != static_cast<uint32_t>(size.x) ||
      frame_image_->Extent().height != static_cast<uint32_t>(size.y)) {
    core_->WaitGPU();
    core_->CreateImage(size.x, size.y, grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &frame_image_);
  }
  std::unique_ptr<grassland::graphics::CommandContext> command_context;
  if (!depth_image_ || depth_image_->Extent().width != static_cast<uint32_t>(size.x) ||
      depth_image_->Extent().height != static_cast<uint32_t>(size.y)) {
    core_->WaitGPU();
    core_->CreateImage(size.x, size.y, grassland::graphics::IMAGE_FORMAT_D32_SFLOAT, &depth_image_);
  }
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(frame_image_.get(), {{0.6, 0.7, 0.8, 1.0}});
  command_context->CmdClearImage(depth_image_.get(), {{1.0f}});
  world_panel_->Render(command_context.get(), frame_image_.get(), depth_image_.get(), view_projection_);
  auto *gui_image = ui_->EndFrame(command_context.get(), frame_image_.get());

  command_context->CmdPresent(window_.get(), gui_image);
  core_->SubmitCommandContext(command_context.get());
}
