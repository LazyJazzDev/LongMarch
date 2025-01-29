#include "grassland/bvh/bvh_cuda.cuh"

namespace grassland {
BVHCuda::BVHCuda(const BVHHost &bvh) {
  nodes_ = bvh.Nodes();
}

BVHRef BVHCuda::GetRef() const {
  return BVHRef{nodes_.data().get()};
}

BVHCuda::operator BVHRef() const {
  return GetRef();
}
}  // namespace grassland
