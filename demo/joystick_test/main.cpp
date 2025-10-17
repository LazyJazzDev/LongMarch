#include "long_march.h"

using namespace long_march;

int main() {
  std::unique_ptr<graphics::Core> core;
  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{}, &core);
  core->InitializeLogicalDeviceAutoSelect(false);

  std::unique_ptr<graphics::Window> window;
  core->CreateWindowObject(1920, 1080, "Joystick Test", &window);

  std::unique_ptr<graphics::Image> color_image;
  core->CreateImage(1920, 1080, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &color_image);

  window->InitImGui(nullptr, 20);

  while (!window->ShouldClose()) {
    window->BeginImGuiFrame();
    ImGui::Begin("Joystick Test", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    for (int js = GLFW_JOYSTICK_1; js <= GLFW_JOYSTICK_LAST; js++) {
      if (glfwJoystickPresent(js)) {
        ImGui::Text("Joystick %d is present", js);
        ImGui::Text("Is gamepad: %s", glfwJoystickIsGamepad(js) ? "Yes" : "No");
        int axis_count;
        const float *axes = glfwGetJoystickAxes(js, &axis_count);
        ImGui::Text("Axes (%d):", axis_count);
        for (int i = 0; i < axis_count; i++) {
          ImGui::Text("  Axis %d: %.3f", i, axes[i]);
        }

        int button_count;
        const unsigned char *buttons = glfwGetJoystickButtons(js, &button_count);
        ImGui::Text("Buttons (%d):", button_count);
        for (int i = 0; i < button_count; i++) {
          ImGui::Text("  Button %d: %s", i, buttons[i] == GLFW_PRESS ? "Pressed" : "Released");
        }

        int hat_count;
        const unsigned char *hats = glfwGetJoystickHats(js, &hat_count);
        ImGui::Text("Hats (%d):", hat_count);
        for (int i = 0; i < hat_count; i++) {
          ImGui::Text("  Hat %d: %d", i, hats[i]);
        }

        const char *name = glfwGetJoystickName(js);
        ImGui::Text("Name: %s", name ? name : "Unknown");
      } else {
        ImGui::Text("Joystick %d is not present", js);
      }
    }

    ImGui::End();
    window->EndImGuiFrame();

    std::unique_ptr<graphics::CommandContext> command_context;
    core->CreateCommandContext(&command_context);
    command_context->CmdPresent(window.get(), color_image.get());
    core->SubmitCommandContext(command_context.get());
    glfwPollEvents();
  }

  return 0;
}
