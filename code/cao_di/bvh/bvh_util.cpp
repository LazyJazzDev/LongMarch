#include "cao_di/bvh/bvh_util.h"

#include <utility>

namespace CD {

AABB Join(const AABB &aabb0, const AABB &aabb1) {
  AABB result;
  result.lower_bound = aabb0.lower_bound.cwiseMin(aabb1.lower_bound);
  result.upper_bound = aabb0.upper_bound.cwiseMax(aabb1.upper_bound);
  return result;
}

BVHNode::BVHNode(AABB aabb, int instance_index) : aabb(std::move(aabb)), instance_index(instance_index) {
  next_node_on_failure = -1;
  lch = -1;
  rch = -1;
}

}  // namespace CD
