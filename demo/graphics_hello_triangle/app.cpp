#include "app.h"

Application::Application() {
  uint32_t glfw_extension_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
  if (!glfw_extensions) {
    const char *glfw_error;
    int ret = glfwGetError(&glfw_error);
    if (ret != GLFW_NO_ERROR) {
      printf("Error: %s\n", glfw_error);
      printf("Is GLFW not initialized: %s\n",
             ret == GLFW_NOT_INITIALIZED ? "yes" : "no");
    }
  }

  glfwInit();
  glfwTerminate();

  grassland::LogInfo("Hello, Triangle!");
}
