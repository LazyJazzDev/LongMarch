#include "long_march.h"
#include "snowberg/gui/gui.h"

using namespace long_march;

int main() {
  std::unique_ptr<graphics::Core> core;
  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{}, &core);
  core->InitializeLogicalDeviceAutoSelect(false);

  std::unique_ptr<graphics::Window> window;
  core->CreateWindowObject(1080, 720, "Joystick Test", &window);
  snowberg::gui::Context ui(core.get(), window.get(), snowberg::gui::DefaultFont());

  std::unique_ptr<graphics::Image> color_image;
  auto resize = [&] {
    const auto size = window->GetFramebufferSize();
    if (size.x > 0 && size.y > 0) {
      core->WaitGPU();
      core->CreateImage(size.x, size.y, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &color_image);
    }
  };
  resize();
  while (!window->ShouldClose()) {
    graphics::Window::PollEvents();
    const auto size = window->GetFramebufferSize();
    if (size.x <= 0 || size.y <= 0)
      continue;
    if (!color_image || color_image->Extent().width != static_cast<uint32_t>(size.x) ||
        color_image->Extent().height != static_cast<uint32_t>(size.y))
      resize();

    ui.BeginFrame();
    ui.BeginPanel("joysticks", 24, 24, 470, "Game controllers");
    bool found = false;
    for (int js = GLFW_JOYSTICK_1; js <= GLFW_JOYSTICK_LAST; ++js) {
      if (!glfwJoystickPresent(js))
        continue;
      found = true;
      const char *name = glfwGetJoystickName(js);
      ui.Heading(name ? name : "Unknown controller");
      ui.Text(glfwJoystickIsGamepad(js) ? "Gamepad: yes" : "Gamepad: no");
      int count = 0;
      const float *axes = glfwGetJoystickAxes(js, &count);
      for (int i = 0; i < count; ++i)
        ui.Text("Axis " + std::to_string(i) + ": " + std::to_string(axes[i]));
      const unsigned char *buttons = glfwGetJoystickButtons(js, &count);
      for (int i = 0; i < count; ++i)
        ui.Text("Button " + std::to_string(i) + ": " + (buttons[i] == GLFW_PRESS ? "pressed" : "released"));
      const unsigned char *hats = glfwGetJoystickHats(js, &count);
      for (int i = 0; i < count; ++i)
        ui.Text("Hat " + std::to_string(i) + ": " + std::to_string(hats[i]));
      ui.Separator();
    }
    if (!found)
      ui.Text("No controller connected");
    ui.EndPanel();

    std::unique_ptr<graphics::CommandContext> commands;
    core->CreateCommandContext(&commands);
    commands->CmdClearImage(color_image.get(), {{0.055f, 0.08f, 0.13f, 1.0f}});
    auto *gui_image = ui.EndFrame(commands.get(), color_image.get());

    commands->CmdPresent(window.get(), gui_image);
    core->SubmitCommandContext(commands.get());
  }
  core->WaitGPU();
  return 0;
}
