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
  core_->CreateImage(window_->GetFramebufferSize().x, window_->GetFramebufferSize().y,
                     grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &frame_image_);
}

void Application::OnClose() {
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
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(frame_image_.get(), {{0.6, 0.7, 0.8, 1.0}});
  auto *gui_image = ui_->EndFrame(command_context.get(), frame_image_.get());

  command_context->CmdPresent(window_.get(), gui_image);
  core_->SubmitCommandContext(command_context.get());
}
