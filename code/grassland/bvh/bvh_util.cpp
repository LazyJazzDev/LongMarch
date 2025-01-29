#include "grassland/bvh/bvh_util.h"

#include <utility>

namespace grassland {

AABB Join(const AABB &aabb0, const AABB &aabb1) {
  AABB result;
  result.lower = aabb0.lower.cwiseMin(aabb1.lower);
  result.upper = aabb0.upper.cwiseMax(aabb1.upper);
  return result;
}

BVHNode::BVHNode(AABB aabb, int instance_index) : aabb(std::move(aabb)), instance_index(instance_index) {
  next_node_on_failure = -1;
  lch = -1;
  rch = -1;
}

}  // namespace grassland
