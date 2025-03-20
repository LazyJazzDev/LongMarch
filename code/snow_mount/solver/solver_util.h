#pragma once
#include "grassland/grassland.h"

namespace snow_mount::solver {

using namespace grassland;

class Core;
class Scene;
struct ObjectPack;
struct ObjectPackView;
struct RigidObject;

struct RigidObjectState {
  Matrix3<float> R;
  Vector3<float> t;
  Vector3<float> v;
  Vector3<float> omega;
  float mass;
  Matrix3<float> inertia;

  LM_DEVICE_FUNC RigidObjectState NextState(float dt) const;
};

struct DirectoryRef {
  const int *first;
  const int *count;
  const int *positions;
};

struct Directory {
  Directory() = default;
  Directory(const std::vector<int> &contents, int num_bucket);
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
