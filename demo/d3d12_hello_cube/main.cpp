#include "d3d12app.h"

int main() {
  auto cocp = GetConsoleOutputCP();
  std::cout << cocp << std::endl;
  SetConsoleOutputCP(CP_UTF8);
  D3D12::Application app;
  app.Run();
  return 0;
}
