#pragma once
#include "grassland/bvh/bvh_util.h"

namespace CD {

class BVHHost {
 public:
  BVHHost() = default;

  BVHHost(const AABB *aabbs, const int *instance_indices, int num_instance);

  void UpdateInstances(const AABB *aabbs, const int *instance_indices, int num_instance);

  BVHRef GetRef() const;

  const std::vector<BVHNode> &Nodes() const;

  operator BVHRef() const;

  void Print() const;

 private:
  std::vector<BVHNode> nodes_;
};

struct BVHHostBuilder {
  BVHNode *nodes;
  int num_nodes;
  int Build(std::pair<AABB, int> *build_contents, int num_contents, int cut_dim, int failure_next_node);
};

}  // namespace CD
