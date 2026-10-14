#pragma once
struct GLFWwindow;

namespace grassland::rhi::backend {

namespace glfw {
bool Initialize();
bool PollEvents();
bool IsWindowThread();
GLFWwindow *CreateWindow(int width, int height, const char *title);
void DestroyWindow(GLFWwindow *window);
bool SetWindowTitle(GLFWwindow *window, const char *title);
}  // namespace glfw

}  // namespace grassland::rhi::backend
