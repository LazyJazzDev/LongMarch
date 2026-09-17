#include "sparkium/backends/offline_cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "sparkium/backends/core/integrator.h"

namespace sparkium::backends {

namespace {

unsigned int ResolveThreadCount() {
  unsigned int count = std::thread::hardware_concurrency();
  if (count == 0)
    count = 1;
  return count;
}

std::string CpuModelName() {
  std::FILE *file = std::fopen("/proc/cpuinfo", "r");
  if (!file)
    return "unknown CPU";
  char line[512];
  std::string model;
  while (std::fgets(line, sizeof(line), file)) {
    if (std::strncmp(line, "model name", 10) == 0) {
      const char *colon = std::strchr(line, ':');
      if (colon) {
        const char *value = colon + 1;
        while (*value == ' ' || *value == '\t')
          ++value;
        model = value;
        while (!model.empty() && (model.back() == '\n' || model.back() == '\r'))
          model.pop_back();
      }
      break;
    }
  }
  std::fclose(file);
  return model.empty() ? std::string("unknown CPU") : model;
}

}  // namespace

uint32_t CpuOfflineBackend::ThreadCount() const {
  return ResolveThreadCount();
}

std::string CpuOfflineBackend::DeviceDescription() const {
  return CpuModelName();
}

void CpuOfflineBackend::Dispatch(const OfflineScene &scene, const RenderSettings &settings, OfflineFilm &film) {
  const DeviceScene &device = scene.Device();
  float4 *color = film.Color();
  float *samples = film.Samples();
  const uint32_t width = film.Width();
  const uint32_t height = film.Height();
  if (width == 0 || height == 0)
    return;

  // One stripe of rows per worker. Every pixel is independent, so the striped
  // partition does not change the result.
  uint32_t workers = std::min<uint32_t>(ResolveThreadCount(), std::max<uint32_t>(height, 1));
  if (workers <= 1) {
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const size_t index = static_cast<size_t>(y) * width + x;
        RenderPixel(device, settings, x, y, width, height, color + index, samples + index);
      }
    }
    return;
  }

  const uint32_t rows_per_worker = (height + workers - 1) / workers;
  std::vector<std::thread> pool;
  pool.reserve(workers);
  for (uint32_t worker = 0; worker < workers; ++worker) {
    const uint32_t begin = worker * rows_per_worker;
    const uint32_t end = std::min(begin + rows_per_worker, height);
    if (begin >= end)
      break;
    pool.emplace_back([&, begin, end]() {
      for (uint32_t y = begin; y < end; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
          const size_t index = static_cast<size_t>(y) * width + x;
          RenderPixel(device, settings, x, y, width, height, color + index, samples + index);
        }
      }
    });
  }
  for (auto &thread : pool)
    thread.join();
}

}  // namespace sparkium::backends

namespace sparkium::backends {

std::unique_ptr<OfflineBackend> CreateCpuBackend() {
  return std::make_unique<CpuOfflineBackend>();
}

}  // namespace sparkium::backends
