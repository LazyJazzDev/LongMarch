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
