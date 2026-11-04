#include <charconv>
#include <fstream>
#include <iostream>

#include "experiment.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  BackendAPI api = BACKEND_API_VULKAN;
  bool sync = true, validation = true;
  uint32_t frames = 8, copies = 32;
  std::string output;
  const auto number = [](const char *text, uint32_t &value) {
    auto end = text + std::char_traits<char>::length(text);
    auto parsed = std::from_chars(text, end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--d3d12")
      api = BACKEND_API_D3D12;
    else if (arg == "--vulkan")
      api = BACKEND_API_VULKAN;
    else if (arg == "--metal")
      api = BACKEND_API_METAL;
    else if (arg == "--without-event") {
      sync = false;
      validation = false;
    } else if (arg == "--no-validation")
      validation = false;
    else if (arg == "--frames" && i + 1 < argc) {
      if (!number(argv[++i], frames))
        return 2;
    } else if (arg == "--copies" && i + 1 < argc) {
      if (!number(argv[++i], copies))
        return 2;
    } else if (arg == "--output" && i + 1 < argc)
      output = argv[++i];
    else {
      std::cerr << "Unknown/incomplete argument: " << arg << '\n';
      return 2;
    }
  }
  if (!frames)
    return 2;
  auto result = multiqueue_demo::Run(api, sync, validation, frames, copies);
  std::cout << "device=" << result.device << " distinct_queues=" << result.distinct_queues << " event=" << sync
            << " frames=" << result.frames << " mismatched_pixels=" << result.mismatched_pixels << '/'
            << uint64_t(result.frames) * 4096 << '\n';
  if (!result.success)
    return 1;
  if (!output.empty()) {
    std::ofstream file(output, std::ios::binary);
    file << "P6\n64 64\n255\n";
    for (size_t i = 0; i < result.pixels.size(); i += 4)
      file.write(reinterpret_cast<const char *>(result.pixels.data() + i), 3);
    if (!file)
      return 1;
  }
  // A race is an observation, not a deterministic failure expectation.
  return sync && result.mismatched_pixels ? 1 : 0;
}
