#include "module.h"

namespace graphics_hello::window {
ModuleWindow::ModuleWindow(grassland::graphics::BackendAPI api, bool headless) : Module(headless) {
  InitializeGraphicsHello(api, core_, false, headless ? 3 : 2);
}

void ModuleWindow::OnInit() {
  if (!headless_)
    core_->CreateWindowObject(1280, 720, GraphicsHelloTitle(core_->API()) + " Graphics Hello Window", &window_);
  for (int i = 0; i < (headless_ ? 3 : 1); ++i)
    core_->CreateImage(1280, 720, grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &color_images_[i]);
  alive_ = true;
}

void ModuleWindow::OnClose() {
  core_->WaitGPU();
  for (auto &image : color_images_)
    image.reset();
}

void ModuleWindow::OnUpdate() {
  if (window_ && window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
}

void ModuleWindow::OnRender() {
  std::unique_ptr<grassland::graphics::CommandContext> context;
  core_->CreateCommandContext(&context);
  auto *image = color_images_[headless_ ? core_->CurrentFrame() : 0].get();
  context->CmdClearImage(image, {{.6f, .7f, .8f, 1}});
  if (!headless_)
    context->CmdPresent(window_.get(), image);
  core_->SubmitCommandContext(context.get());
}

grassland::graphics::Core *ModuleWindow::GetCore() const {
  return core_.get();
}

grassland::graphics::Window *ModuleWindow::GetWindow() const {
  return window_.get();
}

bool ModuleWindow::IsAlive() const {
  return alive_;
}
}  // namespace graphics_hello::window
