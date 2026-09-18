#include "software/layout.hlsli"
RWByteAddressBuffer nodes : register(u0, space0);
ByteAddressBuffer geometries[] : register(t0, space1);
cbuffer BuildParameters : register(b0, space2) {
  uint root;
  uint leaf_count;
  uint primitive_count;
  uint geometry_index;
  uint level_first;
  uint level_count;
  uint sort_stage;
  uint sort_stride;
  uint instance_tree;
};
RWByteAddressBuffer keys : register(u0, space3);
ByteAddressBuffer instances : register(t0, space4);

SoftwareNode EmptyNode() {
  SoftwareNode n;
  n.lo = float3(3.402823e38f, 3.402823e38f, 3.402823e38f);
  n.hi = -n.lo;
  n.first = n.second = SOFTWARE_INVALID;
  return n;
}
void WriteLeaf(uint slot, uint primitive) {
  SoftwareNode n = EmptyNode();
  if (primitive < primitive_count) {
    n.second = primitive;
    if (instance_tree != 0) {
      SoftwareInstance instance = LoadSoftwareInstance(instances, primitive);
      SoftwareNode local = LoadSoftwareNode(nodes, instance.root);
      if (all(local.lo <= local.hi)) {
        for (uint corner = 0; corner < 8; ++corner) {
          float3 p = float3((corner & 1) ? local.hi.x : local.lo.x, (corner & 2) ? local.hi.y : local.lo.y,
                            (corner & 4) ? local.hi.z : local.lo.z);
          p = mul(instance.object_to_world, float4(p, 1));
          n.lo = min(n.lo, p);
          n.hi = max(n.hi, p);
        }
      }
    } else {
      ByteAddressBuffer geometry = geometries[NonUniformResourceIndex(geometry_index)];
      uint position_offset = geometry.Load(8), position_stride = geometry.Load(12);
      uint3 ids = geometry.Load3(geometry.Load(48) + primitive * 12);
      for (uint v = 0; v < 3; ++v) {
        float3 p = LoadFloat3(geometry, position_offset + position_stride * ids[v]);
        n.lo = min(n.lo, p);
        n.hi = max(n.hi, p);
      }
    }
  }
  nodes.Store<SoftwareNode>((root + leaf_count - 1 + slot) * SOFTWARE_NODE_BYTES, n);
}
// Each pass is written as a plain function over one element, with the compute
// entry point kept as a thin wrapper. The CPU backend replays the same passes
// on the host instead of dispatching them.
void InitLeavesKernel(uint index) {
  if (index < leaf_count)
    WriteLeaf(index, index);
}

void ReduceNodesKernel(uint index) {
  if (index >= level_count)
    return;
  index += level_first;
  SoftwareNode n;
  n.first = root + index * 2 + 1;
  n.second = n.first + 1;
  SoftwareNode a = LoadSoftwareNode(nodes, n.first), b = LoadSoftwareNode(nodes, n.second);
  n.lo = min(a.lo, b.lo);
  n.hi = max(a.hi, b.hi);
  nodes.Store<SoftwareNode>((root + index) * SOFTWARE_NODE_BYTES, n);
}

#ifndef SPARKIUM_CPU_SHADER
[numthreads(64, 1, 1)] void InitLeaves(uint3 id : SV_DispatchThreadID) {
  InitLeavesKernel(id.x);
}
[numthreads(64, 1, 1)] void ReduceNodes(uint3 id : SV_DispatchThreadID) {
  ReduceNodesKernel(id.x);
}
#endif

uint SpreadBits(uint v) {
  v = (v | (v << 16)) & 0x030000ff;
  v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3;
  return (v | (v << 2)) & 0x09249249;
}
void MortonKeysKernel(uint index) {
  if (index >= leaf_count)
    return;
  uint code = SOFTWARE_INVALID;
  if (index < primitive_count) {
    SoftwareNode n = LoadSoftwareNode(nodes, root + leaf_count - 1 + index);
    SoftwareNode bounds = LoadSoftwareNode(nodes, root);
    float3 center = (n.lo + n.hi) * 0.5f;
    float3 extent = max(bounds.hi - bounds.lo, 1.0e-20f);
    uint3 p = (uint3)(saturate((center - bounds.lo) / extent) * 1023.0f);
    code = SpreadBits(p.x) | (SpreadBits(p.y) << 1) | (SpreadBits(p.z) << 2);
  }
  keys.Store2(index * 8, uint2(code, index));
}

void BitonicSortKernel(uint i) {
  uint j = i ^ sort_stride;
  if (i >= leaf_count || j <= i)
    return;
  uint2 a = keys.Load2(i * 8), b = keys.Load2(j * 8);
  bool greater = a.x > b.x || (a.x == b.x && a.y > b.y);
  if (greater == ((i & sort_stage) == 0)) {
    keys.Store2(i * 8, b);
    keys.Store2(j * 8, a);
  }
}

void SortLeavesKernel(uint index) {
  if (index < leaf_count)
    WriteLeaf(index, keys.Load(index * 8 + 4));
}

#ifndef SPARKIUM_CPU_SHADER
[numthreads(64, 1, 1)] void MortonKeys(uint3 id : SV_DispatchThreadID) {
  MortonKeysKernel(id.x);
}
[numthreads(64, 1, 1)] void BitonicSort(uint3 id : SV_DispatchThreadID) {
  BitonicSortKernel(id.x);
}
[numthreads(64, 1, 1)] void SortLeaves(uint3 id : SV_DispatchThreadID) {
  SortLeavesKernel(id.x);
}
#endif
