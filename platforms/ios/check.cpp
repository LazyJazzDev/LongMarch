#include <iostream>

#include "RenderSession.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

int main(int argc, char **argv) {
  try {
    if (argc < 4)
      throw std::runtime_error(
          "usage: sparkium_mobile_check <resources> <scene> <output.png> [prepare|replay] [dimension] [spp] [aspect-ratio]");
    RenderSession session(argv[1], argv[2], argc > 5 ? std::stoi(argv[5]) : 128,
                          argc > 4 && std::string(argv[4]) == "prepare", argc > 7 ? std::stod(argv[7]) : 0.0);
    std::vector<uint8_t> pixels;
    int spp = argc > 6 ? std::stoi(argv[6]) : 2;
    if (spp < 1)
      throw std::runtime_error("spp must be positive");
    for (int i = 0; i < spp; ++i)
      pixels = session.Step();
    const int samples = session.Samples();
    auto hdr = session.Display(true);
    float peak = 0;
    for (size_t i = 0; i < hdr.size(); i += 16) {
      float rgb[3];
      std::memcpy(rgb, hdr.data() + i, sizeof(rgb));
      for (float value : rgb) {
        if (!std::isfinite(value))
          throw std::runtime_error("Nonfinite HDR pixel");
        peak = std::max(peak, value);
      }
    }
    auto exposed = session.Display(true, 1);
    float first = 0, second = 0;
    for (size_t i = 0; i < hdr.size(); i += 16) {
      float a, b;
      std::memcpy(&a, hdr.data() + i, 4);
      std::memcpy(&b, exposed.data() + i, 4);
      first += a;
      second += b;
    }
    if (samples != session.Samples() || std::abs(second - 2 * first) > std::max(0.01f, first * 0.001f))
      throw std::runtime_error("Display controls changed film or exposure is incorrect");
    std::cout << "HDR peak " << peak << " (linear sRGB)\n";
    if (!stbi_write_png(argv[3], session.Width(), session.Height(), 4, pixels.data(), session.Width() * 4))
      throw std::runtime_error("cannot write output");
    std::cout << session.Device() << " " << session.Width() << "x" << session.Height() << " " << session.Samples()
              << " spp\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
