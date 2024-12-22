#include "draw_n_gui.h"

int main() {
  DrawNGUI draw_n_gui(grassland::graphics::BACKEND_API_VULKAN);
  draw_n_gui.Run();
}
