#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "RenderSession.h"

int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 7)
      throw std::runtime_error(
          "usage: harmony_scene_check <resources> <scene> [prepare|replay] [edge] [spp] [output.ppm]");
    const int edge = argc > 4 ? std::stoi(argv[4]) : 64;
    const int samples = argc > 5 ? std::stoi(argv[5]) : 1;
    if (edge < 1 || edge > 2048 || samples < 1 || samples > 4096)
      throw std::runtime_error("Invalid check resolution or sample count");
    RenderSession session(argv[1], argv[2], edge, argc < 4 || std::string(argv[3]) == "prepare", 0,
                          grassland::graphics::BACKEND_API_DEFAULT, true, true);
    for (int i = 0; i < samples; ++i)
      session.Render();
    auto pixels = session.Display(true);
    if (session.Samples() != samples || pixels.empty())
      throw std::runtime_error("No rendered sample");
    double rgb_energy = 0;
    for (size_t offset = 0; offset < pixels.size(); offset += sizeof(float)) {
      float value;
      std::memcpy(&value, pixels.data() + offset, sizeof(value));
      if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite HDR pixel");
      if ((offset / sizeof(float)) % 4 != 3)
        rgb_energy += std::abs(value);
    }
    if (rgb_energy <= 0)
      throw std::runtime_error("Rendered scene is entirely black");
    if (argc > 6) {
      const auto sdr = session.Display(false);
      std::ofstream out(argv[6], std::ios::binary);
      out << "P6\n" << session.Width() << " " << session.Height() << "\n255\n";
      for (size_t i = 0; i < sdr.size(); i += 4)
        out.write(reinterpret_cast<const char *>(sdr.data() + i), 3);
      if (!out)
        throw std::runtime_error("Cannot save check image");
    }
    session.Display(false, 1);
    if (session.Samples() != samples)
      throw std::runtime_error("Exposure change altered accumulation");
    std::cout << "PASS " << argv[2] << " " << session.Width() << "x" << session.Height()
              << (session.ComputeFallback() ? " compute fallback" : " ray query") << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
