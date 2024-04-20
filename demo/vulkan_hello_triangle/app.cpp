#include "app.h"

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
  OnShutdown();
}

void Application::OnInit() {
  VkResult result;
  result = long_march::vulkan::CreateInstance(
      long_march::vulkan::InstanceCreateInfo{true}, &instance_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance");
  }

  result = instance_->CreateSurfaceFromGLFWWindow(window_, &surface_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan surface");
  }
}

void Application::OnUpdate() {
}

void Application::OnRender() {
}

void Application::OnShutdown() {
  surface_.reset();
  instance_.reset();
}
