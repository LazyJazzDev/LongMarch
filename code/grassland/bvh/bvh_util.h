#pragma once
#include <Eigen/Eigen>

#include "grassland/util/util.h"

namespace grassland {

struct AABB {
  Eigen::Vector3<float> lower;
  Eigen::Vector3<float> upper;
};

AABB Join(const AABB &aabb0, const AABB &aabb1);

struct BVHNode {
  BVHNode(AABB aabb = {}, int instance_index = 0);

  AABB aabb;
  int instance_index;
  int next_node_on_failure;
  int lch;
  int rch;
};

struct BVHRef {
  const BVHNode *nodes;

  template <typename QueryType, typename ResultType, typename AttachedType>
  LM_DEVICE_FUNC bool Traversal(const QueryType &query,
                                ResultType *result,
                                const AttachedType *attached,
                                bool (*any_hit)(const QueryType &query, const ResultType *result, const AABB &aabb),
                                bool (*instance_hit)(const QueryType &query,
                                                     ResultType *result,
                                                     int instance_index,
                                                     const AttachedType *attached)) const {
    int node_index = 0;
    bool hit = false;
    int test_count = 0;
    while (node_index != -1) {
      test_count++;
      const BVHNode &node = nodes[node_index];
      if (any_hit(query, result, node.aabb)) {
        if (node.lch == -1 && node.rch == -1) {
          bool inst_hit = instance_hit(query, result, node.instance_index, attached);
          hit |= inst_hit;
          node_index = node.next_node_on_failure;
        } else {
          node_index = node.lch;
        }
      } else {
        node_index = node.next_node_on_failure;
      }
    }
    // printf("Test count: %d\n", test_count);
    return hit;
  }
};

}  // namespace grassland
