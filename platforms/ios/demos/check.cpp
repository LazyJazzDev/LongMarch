#include <algorithm>
#include <cmath>
#include <iostream>

#include "DemoSession.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
int main(int argc, char **argv) {
  try {
    if (argc < 4)
      throw std::runtime_error("usage: mobile_demo_check <resources> <demo|all> <output-dir> [prepare]");
    std::filesystem::create_directories(argv[3]);
    std::vector<std::string> names =
        std::string(argv[2]) == "all" ? DemoSession::Names() : std::vector<std::string>{argv[2]};
    for (const auto &name : names) {
      DemoSession session(argv[1], name, argc > 4 && std::string(argv[4]) == "prepare");
      session.Configure(4096, 10, .03f, true, 0, 0, 0);
      auto before = session.Positions();
      session.Render();
      if (name == "nbody_cs") {
        auto after = session.Positions();
        if (before == after)
          throw std::runtime_error("NBody positions did not move");
        for (auto p : after)
          if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            throw std::runtime_error("Nonfinite particle");
        session.Configure(4096, 10, .03f, false, 0, 0, 0);
        session.Render();
        if (session.Positions() != after)
          throw std::runtime_error("Paused simulation moved");
        session.Configure(4096, 10, .03f, false, 0, 0, 1);
        auto first_reset = session.Positions();
        if (first_reset == before || first_reset == after)
          throw std::runtime_error("Reset did not generate a new particle distribution");
        session.Configure(4096, 10, .03f, false, 0, 0, 1);
        session.Render();
        if (session.Positions() != first_reset)
          throw std::runtime_error("Unchanged settings reset the paused simulation");
        session.Configure(4096, 10, .03f, false, 0, 0, 2);
        auto second_reset = session.Positions();
        if (second_reset == first_reset || second_reset == before)
          throw std::runtime_error("Consecutive resets reused a particle distribution");
        session.Render();
        if (session.Positions() != second_reset)
          throw std::runtime_error("Reset resumed the paused simulation");
      }
      if (name == "graphics_hello_resize") {
        session.Resize(900, 600);
        session.Render();
      }
      auto extent = session.Image()->Extent();
      std::vector<float> pixels(extent.width * extent.height * 4);
      session.Image()->DownloadData(pixels.data());
      std::vector<uint8_t> bytes(pixels.size());
      for (size_t i = 0; i < pixels.size(); ++i) {
        if (!std::isfinite(pixels[i]))
          throw std::runtime_error("Nonfinite pixel");
        bytes[i] = uint8_t(std::clamp(pixels[i], 0.f, 1.f) * 255.f + .5f);
      }
      auto output = std::filesystem::path(argv[3]) / (name + ".png");
      if (!stbi_write_png(output.c_str(), extent.width, extent.height, 4, bytes.data(), extent.width * 4))
        throw std::runtime_error("Cannot save image");
      std::cout << "PASS " << name << " " << extent.width << "x" << extent.height << " GPU "
                << session.GPUMilliseconds() << " ms\n";
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
