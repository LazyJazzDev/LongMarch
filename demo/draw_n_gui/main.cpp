#include "draw_n_gui.h"

int main() {
  DrawNGUI draw_n_gui(grassland::graphics::BACKEND_API_D3D12);
  draw_n_gui.Run();
}
