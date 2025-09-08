#include "app.h"

int main() {
  Application app{CD::graphics::BACKEND_API_DEFAULT};

  app.OnInit();

  while (app.IsAlive()) {
    if (app.IsAlive()) {
      app.OnUpdate();
    }
    if (app.IsAlive()) {
      app.OnRender();
    }

    glfwPollEvents();
  }

  app.OnClose();

  return 0;
}
