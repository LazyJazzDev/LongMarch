#include "nbody_cuda.h"

using namespace long_march;

void PrintArgHelper() {
  printf("Usage: demo_nbody_cuda [-headless] [-nstep <num_steps>] [-device <device_id>]\n");
  printf("Options:\n");
  printf("  -headless           Run in headless mode (no GUI)\n");
  printf("  -nstep <num_steps>  Number of simulation steps to run (default: 200)\n");
  printf("  -device <device_id> CUDA device ID to use (default: 0)\n");
}

int main(int argc, char *argv[]) {
  NBodyCUDA::Settings settings{};

  for (int head = 0; head < argc; head++) {
    std::string arg = argv[head];
    if (arg == "-headless") {
      settings.headless = true;
    } else if (arg == "-nstep") {
      if (head + 1 >= argc) {
        PrintArgHelper();
        return 1;
      }
      settings.num_step = std::atoi(argv[++head]);
    } else if (arg == "-device") {
      if (head + 1 >= argc) {
        PrintArgHelper();
        return 1;
      }
      settings.device_id = std::atoi(argv[++head]);
    } else {
      PrintArgHelper();
      return 1;
    }
  }

  NBodyCUDA app(settings);
  app.Run();
}
