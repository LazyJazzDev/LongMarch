#pragma once
#include <thrust/device_vector.h>

#include "grassland/bvh/bvh_host.h"
#include "grassland/bvh/bvh_util.h"

namespace CD {

class BVHCuda {
 public:
  BVHCuda() {
  }

  BVHCuda(const BVHHost &bvh);

  BVHRef GetRef() const;

  operator BVHRef() const;

 private:
  thrust::device_vector<BVHNode> nodes_;
};

}  // namespace CD
