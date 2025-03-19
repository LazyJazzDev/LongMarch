#pragma once
#include "grassland/grassland.h"

namespace snow_mount::solver {

using namespace grassland;

class Core;
class Scene;
struct ObjectPack;
struct ObjectPackView;
struct RigidObject;

struct DirectoryRef {
  const int *first;
  const int *count;
  const int *positions;
};

struct Directory {
  Directory(const std::vector<int> &contents);
  std::vector<int> first;
  std::vector<int> count;
  std::vector<int> positions;
  operator DirectoryRef() const;
};

#if defined(__CUDACC__)
struct DirectoryDevice {
  DirectoryDevice() = default;
  DirectoryDevice(const Directory &directory);
  thrust::device_vector<int> first;
  thrust::device_vector<int> count;
  thrust::device_vector<int> positions;
  operator DirectoryRef() const;
};
#endif

}  // namespace snow_mount::solver
