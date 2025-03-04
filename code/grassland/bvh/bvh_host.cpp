#include "grassland/bvh/bvh_host.h"

namespace grassland {

BVHHost::BVHHost(const AABB *aabbs, const int *instance_indices, int num_instance) {
  UpdateInstances(aabbs, instance_indices, num_instance);
}

void BVHHost::UpdateInstances(const AABB *aabbs, const int *instance_indices, int num_instance) {
  if (nodes_.size() != num_instance * 2 - 1) {
    nodes_.resize(num_instance * 2 - 1);
  }
  std::vector<std::pair<AABB, int>> contents(num_instance);
  for (int i = 0; i < num_instance; i++) {
    contents[i] = {aabbs[i], instance_indices[i]};
  }
  BVHHostBuilder builder{nodes_.data(), 0};
  builder.Build(contents.data(), contents.size(), 0, -1);
}

BVHRef BVHHost::GetRef() const {
  return BVHRef{nodes_.data()};
}

const std::vector<BVHNode> &BVHHost::Nodes() const {
  return nodes_;
}

BVHHost::operator BVHRef() const {
  return GetRef();
}

void BVHHost::Print() const {
  printf("BVH nodes:\n");
  for (size_t i = 0; i < nodes_.size(); i++) {
    const BVHNode &node = nodes_[i];
    printf("Node %zd: instance_index = %d, lch = %d, rch = %d, next_node_on_failure = %d\n", i, node.instance_index,
           node.lch, node.rch, node.next_node_on_failure);
    printf("  AABB: lower = {%f %f %f}, upper = {%f %f %f}\n", node.aabb.lower_bound[0], node.aabb.lower_bound[1],
           node.aabb.lower_bound[2], node.aabb.upper_bound[0], node.aabb.upper_bound[1], node.aabb.upper_bound[2]);
  }
}

int BVHHostBuilder::Build(std::pair<AABB, int> *build_contents, int num_contents, int cut_dim, int failure_next_node) {
  int node_index = num_nodes++;
  BVHNode &node = nodes[node_index];
  const int next_dim = (cut_dim + 1) % 3;
  if (num_contents == 1) {
    node = BVHNode{build_contents[0].first, build_contents[0].second};
  } else {
    std::sort(build_contents, build_contents + num_contents,
              [cut_dim](const std::pair<AABB, int> &a, const std::pair<AABB, int> &b) {
                return a.first.lower_bound[cut_dim] < b.first.lower_bound[cut_dim];
              });
    int mid = num_contents / 2;
    int rch = Build(build_contents + mid, num_contents - mid, next_dim, failure_next_node);
    int lch = Build(build_contents, mid, next_dim, rch);
    node.lch = lch;
    node.rch = rch;
    node.aabb = Join(nodes[node.lch].aabb, nodes[node.rch].aabb);
    node.instance_index = -1;
  }
  node.next_node_on_failure = failure_next_node;
  return node_index;
}

}  // namespace grassland
