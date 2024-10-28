#include "d3d12app.h"

int main() {
  auto cocp = GetConsoleOutputCP();
  SetConsoleOutputCP(CP_UTF8);
  std::cout << cocp << std::endl;
  d3d12::Application app;
  app.Run();
  return 0;
}
