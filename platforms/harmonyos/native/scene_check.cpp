#include <cmath>
#include <cstring>
#include <iostream>

#include "RenderSession.h"

int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 4)
      throw std::runtime_error("usage: harmony_scene_check <resources> <scene> [prepare|replay]");
    RenderSession session(argv[1], argv[2], 64, argc < 4 || std::string(argv[3]) == "prepare", 0,
                          grassland::graphics::BACKEND_API_DEFAULT, true, true);
    session.Render();
    auto pixels = session.Display(true);
    if (session.Samples() != 1 || pixels.empty())
      throw std::runtime_error("No rendered sample");
    for (size_t offset = 0; offset < pixels.size(); offset += sizeof(float)) {
      float value;
      std::memcpy(&value, pixels.data() + offset, sizeof(value));
      if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite HDR pixel");
    }
    session.Display(false, 1);
    if (session.Samples() != 1)
      throw std::runtime_error("Exposure change altered accumulation");
    std::cout << "PASS " << argv[2] << " " << session.Width() << "x" << session.Height()
              << (session.ComputeFallback() ? " compute fallback" : " ray query") << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
