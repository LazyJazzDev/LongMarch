#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <random>

namespace sparkium_test {
// Temporary files are deliberately removed before rendering in the tests below.
class SceneFiles {
 public:
  SceneFiles() {
    directory = std::filesystem::path(SPARKIUM_TEST_TEMP_DIR) /
                ("sparkium-memory-scene-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(directory);
    std::ofstream(directory / "mesh.obj") << "v -2 -2 0\nv 2 -2 0\nv 0 2 0\n"
                                             "vt 0 0\nvt 1 0\nvt 0.5 1\nf 1/1 2/2 3/3\n";
    // A 1x1 uncompressed TGA texel. No generated binary is checked into Git.
    std::array<unsigned char, 21> tga{};
    tga[2] = 2;
    tga[12] = tga[14] = 1;
    tga[16] = 24;
    tga[18] = 64;
    tga[19] = 128;
    tga[20] = 255;
    std::ofstream texture(directory / "texture.tga", std::ios::binary);
    texture.write(reinterpret_cast<const char *>(tga.data()), tga.size());
    std::ofstream(directory / "scene.json") << R"({
      "format":"sparkium-scene","version":1,"name":"Memory scene",
      "film":{"width":17,"height":13,"view_transform":"standard"},
      "renderer":{"pipeline":"auto","samples_per_dispatch":2,"max_bounces":2,
                  "background_color":[0.1,0.2,0.3]},
      "camera":{"eye":[0,0,4],"target":[0,0,0]},
      "materials":{
        "surface":{"type":"principled","textures":{"base_color":"texture.tga","emission":"texture.tga"}},
        "graph":{"type":"shader_graph","graph":{
          "nodes":{"image":{"type":"image_texture","path":"texture.tga"}},
          "surface":{"base_color":{"node":"image"},"emission":{"node":"image"}}}}
      },
      "geometries":{"triangle":{"type":"mesh","path":"mesh.obj","generate_normals":true}},
      "entities":[{"type":"mesh","geometry":"triangle","material":"graph"}]
    })";
  }

  void RemoveSources() {
    // Delete only the three exact files created by this fixture.
    for (const auto *name : {"scene.json", "mesh.obj", "texture.tga"})
      std::filesystem::remove(directory / name);
  }

  ~SceneFiles() {
    std::error_code error;
    for (const auto *name : {"scene.json", "mesh.obj", "texture.tga"})
      std::filesystem::remove(directory / name, error);
    std::filesystem::remove(directory, error);
  }

  std::filesystem::path directory;
};
}  // namespace sparkium_test
