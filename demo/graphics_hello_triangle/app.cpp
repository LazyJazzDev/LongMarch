#include "app.h"

Application::Application() {
  glfwInit();
  glfwTerminate();
  grassland::LogInfo("Hello, Triangle!");
}
