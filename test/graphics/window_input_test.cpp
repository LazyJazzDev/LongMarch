#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "grassland/graphics/window.h"

namespace {
using grassland::graphics::Window;

class TestWindow : public Window {
 public:
  TestWindow() : Window(320, 240, "Window input test", false, true, false) {
  }

  void InitImGui(const char *, float) override {
  }

  void TerminateImGui() override {
  }

  void BeginImGuiFrame() override {
  }

  void EndImGuiFrame() override {
  }

  ImGuiContext *GetImGuiContext() const override {
    return nullptr;
  }
};

TEST(WindowInput, NativeCallbacksRouteToTheirWindowAndSupportMultipleSubscribers) {
  TestWindow first;
  TestWindow second;
  Window::PollEvents();
  std::vector<bool> first_entries, second_entries, focus_changes;
  std::vector<glm::ivec2> framebuffer_sizes;
  int observer_calls = 0;
  const auto first_id =
      first.CursorEnterEvent().RegisterCallback([&](bool entered) { first_entries.push_back(entered); });
  const auto second_id =
      second.CursorEnterEvent().RegisterCallback([&](bool entered) { second_entries.push_back(entered); });
  const auto observer_id = first.CursorEnterEvent().RegisterCallback([&](bool) { ++observer_calls; });
  const auto focus_id = first.FocusEvent().RegisterCallback([&](bool focused) { focus_changes.push_back(focused); });
  const auto frame_id =
      first.FramebufferResizeEvent().RegisterCallback([&](int w, int h) { framebuffer_sizes.emplace_back(w, h); });

  // Retrieve and restore the installed GLFW bridges, then inject native callback
  // arguments. This verifies routing without depending on desktop cursor/focus.
  auto enter = glfwSetCursorEnterCallback(first.GLFWWindow(), nullptr);
  glfwSetCursorEnterCallback(first.GLFWWindow(), enter);
  auto focus = glfwSetWindowFocusCallback(first.GLFWWindow(), nullptr);
  glfwSetWindowFocusCallback(first.GLFWWindow(), focus);
  auto framebuffer = glfwSetFramebufferSizeCallback(first.GLFWWindow(), nullptr);
  glfwSetFramebufferSizeCallback(first.GLFWWindow(), framebuffer);
  ASSERT_NE(enter, nullptr);
  ASSERT_NE(focus, nullptr);
  ASSERT_NE(framebuffer, nullptr);
  enter(first.GLFWWindow(), GLFW_TRUE);
  enter(second.GLFWWindow(), GLFW_FALSE);
  enter(first.GLFWWindow(), GLFW_FALSE);
  focus(first.GLFWWindow(), GLFW_FALSE);
  focus(first.GLFWWindow(), GLFW_TRUE);
  framebuffer(first.GLFWWindow(), 640, 480);
  framebuffer(first.GLFWWindow(), 0, 0);
  EXPECT_EQ(first_entries, (std::vector<bool>{true, false}));
  EXPECT_EQ(second_entries, (std::vector<bool>{false}));
  EXPECT_EQ(focus_changes, (std::vector<bool>{false, true}));
  EXPECT_EQ(framebuffer_sizes, (std::vector<glm::ivec2>{{640, 480}, {0, 0}}));
  EXPECT_EQ(observer_calls, 2);

  first.CursorEnterEvent().UnregisterCallback(first_id);
  enter(first.GLFWWindow(), GLFW_TRUE);
  EXPECT_EQ(first_entries.size(), 2);
  EXPECT_EQ(observer_calls, 3);
  first.CursorEnterEvent().UnregisterCallback(observer_id);
  first.FocusEvent().UnregisterCallback(focus_id);
  first.FramebufferResizeEvent().UnregisterCallback(frame_id);
  first.CloseWindow();
  enter(second.GLFWWindow(), GLFW_TRUE);
  EXPECT_EQ(second_entries, (std::vector<bool>{false, true}));
  second.CursorEnterEvent().UnregisterCallback(second_id);
}

TEST(WindowInput, RealResizeReportsPhysicalPixelsAndCloseRequestKeepsWindowAlive) {
  TestWindow window;
  Window::PollEvents();
  glm::ivec2 logical{0}, physical{0};
  const auto resize_id = window.ResizeEvent().RegisterCallback([&](int w, int h) { logical = {w, h}; });
  const auto framebuffer_id =
      window.FramebufferResizeEvent().RegisterCallback([&](int w, int h) { physical = {w, h}; });
  window.Resize(400, 300);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while ((logical.x == 0 || physical.x == 0) && std::chrono::steady_clock::now() < deadline) {
    Window::PollEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_EQ(logical, window.GetSize());
  EXPECT_EQ(logical, glm::ivec2(400, 300));
  EXPECT_EQ(physical, window.GetFramebufferSize());
  EXPECT_GT(physical.x, 0);
  EXPECT_GT(physical.y, 0);
  window.RequestClose();
  EXPECT_TRUE(window.ShouldClose());
  EXPECT_NE(window.GLFWWindow(), nullptr);
  EXPECT_EQ(window.GetSize(), glm::ivec2(400, 300));
  window.ResizeEvent().UnregisterCallback(resize_id);
  window.FramebufferResizeEvent().UnregisterCallback(framebuffer_id);
}
}  // namespace
