#include "sparkium/pipelines/raytracing/cpu/bvh_builder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace sparkium::raytracing::cpu {
namespace {

constexpr int kBinCount = 16;
// The shader traverses with a 32-entry stack, so the tree must stay shallower
// than that. SAH is skipped past this depth and a median split is used instead,
// which also keeps the build O(n log n) on degenerate inputs.
constexpr int kMaxSahDepth = 24;

struct Bounds {
  float lo[3];
  float hi[3];
};

Bounds Empty() {
  return {{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
           std::numeric_limits<float>::max()},
          {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
           -std::numeric_limits<float>::max()}};
}

void Expand(Bounds &bounds, const BvhPrimitive &primitive) {
  for (int axis = 0; axis < 3; ++axis) {
    bounds.lo[axis] = std::min(bounds.lo[axis], primitive.lo[axis]);
    bounds.hi[axis] = std::max(bounds.hi[axis], primitive.hi[axis]);
  }
}

float Extent(const Bounds &bounds, int axis) {
  return bounds.hi[axis] - bounds.lo[axis];
}

float Area(const Bounds &bounds) {
  long double area = 0.0L;
  for (int axis = 0; axis < 3; ++axis) {
    const int other = (axis + 1) % 3;
    const int last = (axis + 2) % 3;
    // Degenerate (flat) axes contribute nothing, matching the usual convention
    // of treating the surface area of a zero-thickness box as the area of the
    // two faces that survive.
    area += static_cast<long double>(Extent(bounds, other)) * static_cast<long double>(Extent(bounds, last));
  }
  return static_cast<float>(area * 2.0L);
}

float SurfaceArea(const std::vector<BvhPrimitive> &primitives, uint32_t begin, uint32_t end) {
  Bounds bounds = Empty();
  for (uint32_t i = begin; i < end; ++i)
    Expand(bounds, primitives[i]);
  return Area(bounds);
}

// Binned SAH over one axis: returns the candidate split, or false when the
// axis offers nothing better than leaving the range as it is.
// A split is identified by the axis it partitions along and the first bin that
// belongs to the right side. The caller re-derives the binning with the same
// centroid bounds and scale, so partitioning by `bin < first_right_bin` gives
// exactly the two sets the cost was computed from.
struct SplitCandidate {
  bool found{false};
  int axis{0};
  uint32_t first_right_bin{0};
  float cost{std::numeric_limits<float>::max()};
};

struct Bin {
  Bounds bounds{Empty()};
  uint32_t count{0};
};

SplitCandidate FindSplit(const std::vector<BvhPrimitive> &primitives, uint32_t begin, uint32_t end) {
  Bounds centroid_bounds = Empty();
  for (uint32_t i = begin; i < end; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float centroid = 0.5f * (primitives[i].lo[axis] + primitives[i].hi[axis]);
      centroid_bounds.lo[axis] = std::min(centroid_bounds.lo[axis], centroid);
      centroid_bounds.hi[axis] = std::max(centroid_bounds.hi[axis], centroid);
    }
  }

  const float parent_area = SurfaceArea(primitives, begin, end);
  const uint32_t count = end - begin;
  SplitCandidate best;

  for (int axis = 0; axis < 3; ++axis) {
    const float extent = Extent(centroid_bounds, axis);
    if (!(extent > 0.0f))
      continue;
    const float scale = static_cast<float>(kBinCount) * (1.0f - 1.0e-6f) / extent;

    std::array<Bin, kBinCount> bins{};
    for (uint32_t i = begin; i < end; ++i) {
      const float centroid = 0.5f * (primitives[i].lo[axis] + primitives[i].hi[axis]);
      const int index = std::clamp(static_cast<int>(scale * (centroid - centroid_bounds.lo[axis])), 0, kBinCount - 1);
      bins[index].count++;
      Expand(bins[index].bounds, primitives[i]);
    }

    // One sweep from each end gives the bounds and counts of every prefix and
    // suffix, which is what the cost of a split needs.
    std::array<Bounds, kBinCount> left_bounds;
    std::array<uint32_t, kBinCount> left_count{};
    Bounds running = Empty();
    uint32_t running_count = 0;
    for (int i = 0; i < kBinCount; ++i) {
      if (i == 0)
        running = bins[i].bounds;
      else
        for (int component = 0; component < 3; ++component) {
          running.lo[component] = std::min(running.lo[component], bins[i].bounds.lo[component]);
          running.hi[component] = std::max(running.hi[component], bins[i].bounds.hi[component]);
        }
      running_count += bins[i].count;
      left_bounds[i] = running;
      left_count[i] = running_count;
    }
    std::array<Bounds, kBinCount> right_bounds;
    std::array<uint32_t, kBinCount> right_count{};
    running = Empty();
    running_count = 0;
    for (int i = kBinCount - 1; i >= 0; --i) {
      if (i == kBinCount - 1)
        running = bins[i].bounds;
      else
        for (int component = 0; component < 3; ++component) {
          running.lo[component] = std::min(running.lo[component], bins[i].bounds.lo[component]);
          running.hi[component] = std::max(running.hi[component], bins[i].bounds.hi[component]);
        }
      running_count += bins[i].count;
      right_bounds[i] = running;
      right_count[i] = running_count;
    }

    for (int split = 0; split < kBinCount - 1; ++split) {
      if (left_count[split] == 0 || right_count[split + 1] == 0)
        continue;
      const float cost = Area(left_bounds[split]) * static_cast<float>(left_count[split]) +
                         Area(right_bounds[split + 1]) * static_cast<float>(right_count[split + 1]);
      if (cost < best.cost) {
        best.found = true;
        best.axis = axis;
        best.first_right_bin = static_cast<uint32_t>(split) + 1;
        best.cost = cost;
      }
    }
  }

  // Every leaf holds exactly one primitive here, so a split can never be
  // declined on the grounds that the range would make a cheaper leaf; the cost
  // test above only guards against splitting on a degenerate axis. But a split
  // that does not reduce the summed area still means the bins are useless, and
  // the caller falls back to a median split in that case.
  if (best.found && !(best.cost < parent_area * static_cast<float>(count))) {
    best.found = false;
  }
  return best;
}

}  // namespace

void BuildBvh(std::vector<SoftwareNode> &nodes, uint32_t root, std::vector<BvhPrimitive> &primitives) {
  const uint32_t count = static_cast<uint32_t>(primitives.size());
  if (count == 0)
    return;

  uint32_t next = root;

  // Depth-first allocation: the caller reserves BvhNodeCount(count) nodes and
  // this fills them in index order, so a node's children are simply the next
  // indices written.
  struct Frame {
    uint32_t begin;
    uint32_t end;
    uint32_t node;
    int depth;
  };

  std::vector<Frame> stack;
  stack.push_back({0, count, next++, 0});

  while (!stack.empty()) {
    const Frame frame = stack.back();
    stack.pop_back();

    Bounds bounds = Empty();
    for (uint32_t i = frame.begin; i < frame.end; ++i)
      Expand(bounds, primitives[i]);

    if (frame.end - frame.begin == 1) {
      SoftwareNode &node = nodes[frame.node];
      std::copy(bounds.lo, bounds.lo + 3, node.lo);
      std::copy(bounds.hi, bounds.hi + 3, node.hi);
      node.first = kSoftwareInvalid;
      node.second = primitives[frame.begin].index;
      continue;
    }

    SplitCandidate split = frame.depth < kMaxSahDepth ? FindSplit(primitives, frame.begin, frame.end)
                                                      : SplitCandidate{};

    uint32_t mid = frame.begin;
    if (split.found) {
      // Partition on the same centroid binning the sweep costed.
      const int axis = split.axis;
      Bounds centroid_bounds = Empty();
      for (uint32_t i = frame.begin; i < frame.end; ++i) {
        const float centroid = 0.5f * (primitives[i].lo[axis] + primitives[i].hi[axis]);
        centroid_bounds.lo[axis] = std::min(centroid_bounds.lo[axis], centroid);
        centroid_bounds.hi[axis] = std::max(centroid_bounds.hi[axis], centroid);
      }
      const float extent = Extent(centroid_bounds, axis);
      const float scale = extent > 0.0f ? static_cast<float>(kBinCount) * (1.0f - 1.0e-6f) / extent : 0.0f;
      auto bin_of = [&](const BvhPrimitive &primitive) {
        const float centroid = 0.5f * (primitive.lo[axis] + primitive.hi[axis]);
        return static_cast<uint32_t>(
            std::clamp(static_cast<int>(scale * (centroid - centroid_bounds.lo[axis])), 0, kBinCount - 1));
      };
      auto middle =
          std::partition(primitives.begin() + frame.begin, primitives.begin() + frame.end,
                         [&](const BvhPrimitive &primitive) { return bin_of(primitive) < split.first_right_bin; });
      mid = static_cast<uint32_t>(middle - primitives.begin());
    }

    if (mid == frame.begin || mid == frame.end) {
      // Either SAH declined the range or the partition landed entirely on one
      // side, which happens when many centroids share a bin. Split at the
      // median of the widest centroid axis so the tree stays balanced.
      int axis = 0;
      float widest = -1.0f;
      for (int candidate = 0; candidate < 3; ++candidate) {
        float lo = std::numeric_limits<float>::max(), hi = -std::numeric_limits<float>::max();
        for (uint32_t i = frame.begin; i < frame.end; ++i) {
          const float centroid = 0.5f * (primitives[i].lo[candidate] + primitives[i].hi[candidate]);
          lo = std::min(lo, centroid);
          hi = std::max(hi, centroid);
        }
        if (hi - lo > widest) {
          widest = hi - lo;
          axis = candidate;
        }
      }
      const uint32_t middle = frame.begin + (frame.end - frame.begin) / 2;
      std::nth_element(primitives.begin() + frame.begin, primitives.begin() + middle, primitives.begin() + frame.end,
                       [axis](const BvhPrimitive &a, const BvhPrimitive &b) {
                         return 0.5f * (a.lo[axis] + a.hi[axis]) < 0.5f * (b.lo[axis] + b.hi[axis]);
                       });
      mid = middle;
    }

    SoftwareNode &node = nodes[frame.node];
    std::copy(bounds.lo, bounds.lo + 3, node.lo);
    std::copy(bounds.hi, bounds.hi + 3, node.hi);
    node.first = next++;
    node.second = next++;

    // Pushed right first so the left child, which is visited first by the
    // shader's stack, is written to the lower index.
    stack.push_back({mid, frame.end, node.second, frame.depth + 1});
    stack.push_back({frame.begin, mid, node.first, frame.depth + 1});
  }
}

}  // namespace sparkium::raytracing::cpu
