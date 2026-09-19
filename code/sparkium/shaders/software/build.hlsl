#include "native_contract.hlsli"
#include "software/layout.hlsli"
SP_RESOURCE(RWByteAddressBuffer, nodes, u0, 0);
#define SP_BINDING_nodes SP_RESOURCE_ACCESS(RWByteAddressBuffer, nodes, 0)
SP_ARRAY_RESOURCE(ByteAddressBuffer, geometries, t0, 1);
#define SP_BINDING_geometries SP_ARRAY_ACCESS(ByteAddressBuffer, geometries, 1)

struct BuildParameters {
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

SP_RESOURCE(ConstantBuffer<BuildParameters>, build_parameters, b0, 2);
#define SP_BINDING_build_parameters SP_RESOURCE_ACCESS(ConstantBuffer<BuildParameters>, build_parameters, 2)
SP_RESOURCE(RWByteAddressBuffer, keys, u0, 3);
#define SP_BINDING_keys SP_RESOURCE_ACCESS(RWByteAddressBuffer, keys, 3)
SP_RESOURCE(ByteAddressBuffer, instances, t0, 4);
#define SP_BINDING_instances SP_RESOURCE_ACCESS(ByteAddressBuffer, instances, 4)

SoftwareNode EmptyNode() {
  SoftwareNode n;
  n.lo = float3(3.402823e38f, 3.402823e38f, 3.402823e38f);
  n.hi = -n.lo;
  n.first = n.second = SOFTWARE_INVALID;
  return n;
}

void WriteLeaf(SP_CONTEXT uint slot, uint primitive) {
  SoftwareNode n = EmptyNode();
  if (primitive < SP_BINDING_build_parameters.primitive_count) {
    n.second = primitive;
    if (SP_BINDING_build_parameters.instance_tree != 0) {
      SoftwareInstance instance = LoadSoftwareInstance(SP_BINDING_instances, primitive);
      SoftwareNode local = LoadSoftwareNode(SP_BINDING_nodes, instance.root);
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
      ByteAddressBuffer geometry = SP_BINDING_geometries[SP_NONUNIFORM(SP_BINDING_build_parameters.geometry_index)];
      uint position_offset = geometry.Load(8), position_stride = geometry.Load(12);
      uint3 ids = geometry.Load3(geometry.Load(48) + primitive * 12);
      for (uint v = 0; v < 3; ++v) {
        float3 p = LoadFloat3(geometry, position_offset + position_stride * ids[v]);
        n.lo = min(n.lo, p);
        n.hi = max(n.hi, p);
      }
    }
  }

  SP_BINDING_nodes.Store<SoftwareNode>(
      (SP_BINDING_build_parameters.root + SP_BINDING_build_parameters.leaf_count - 1 + slot) * SOFTWARE_NODE_BYTES, n);
}

SP_NUMTHREADS(64, 1, 1) void InitLeaves(SP_CONTEXT uint3 id : SV_DispatchThreadID) {
  if (id.x < SP_BINDING_build_parameters.leaf_count)
    WriteLeaf(SP_CONTEXT_ARG id.x, id.x);
}

SP_NUMTHREADS(64, 1, 1) void ReduceNodes(SP_CONTEXT uint3 id : SV_DispatchThreadID) {
  if (id.x >= SP_BINDING_build_parameters.level_count)
    return;
  uint index = SP_BINDING_build_parameters.level_first + id.x;
  SoftwareNode n;
  n.first = SP_BINDING_build_parameters.root + index * 2 + 1;
  n.second = n.first + 1;
  SoftwareNode a = LoadSoftwareNode(SP_BINDING_nodes, n.first), b = LoadSoftwareNode(SP_BINDING_nodes, n.second);
  n.lo = min(a.lo, b.lo);
  n.hi = max(a.hi, b.hi);
  SP_BINDING_nodes.Store<SoftwareNode>((SP_BINDING_build_parameters.root + index) * SOFTWARE_NODE_BYTES, n);
}

uint SpreadBits(uint v) {
  v = (v | (v << 16)) & 0x030000ff;
  v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3;
  return (v | (v << 2)) & 0x09249249;
}

SP_NUMTHREADS(64, 1, 1) void MortonKeys(SP_CONTEXT uint3 id : SV_DispatchThreadID) {
  if (id.x >= SP_BINDING_build_parameters.leaf_count)
    return;
  uint code = SOFTWARE_INVALID;
  if (id.x < SP_BINDING_build_parameters.primitive_count) {
    SoftwareNode n = LoadSoftwareNode(
        SP_BINDING_nodes, SP_BINDING_build_parameters.root + SP_BINDING_build_parameters.leaf_count - 1 + id.x);
    SoftwareNode bounds = LoadSoftwareNode(SP_BINDING_nodes, SP_BINDING_build_parameters.root);
    float3 center = (n.lo + n.hi) * 0.5f;
    float3 extent = max(bounds.hi - bounds.lo, 1.0e-20f);
    uint3 p = (uint3)(saturate((center - bounds.lo) / extent) * 1023.0f);
    code = SpreadBits(p.x) | (SpreadBits(p.y) << 1) | (SpreadBits(p.z) << 2);
  }

  SP_BINDING_keys.Store2(id.x * 8, uint2(code, id.x));
}

SP_NUMTHREADS(64, 1, 1) void BitonicSort(SP_CONTEXT uint3 id : SV_DispatchThreadID) {
  uint i = id.x, j = i ^ SP_BINDING_build_parameters.sort_stride;
  if (i >= SP_BINDING_build_parameters.leaf_count || j <= i)
    return;
  uint2 a = SP_BINDING_keys.Load2(i * 8), b = SP_BINDING_keys.Load2(j * 8);
  bool greater = a.x > b.x || (a.x == b.x && a.y > b.y);
  if (greater == ((i & SP_BINDING_build_parameters.sort_stage) == 0)) {
    SP_BINDING_keys.Store2(i * 8, b);
    SP_BINDING_keys.Store2(j * 8, a);
  }
}

SP_NUMTHREADS(64, 1, 1) void SortLeaves(SP_CONTEXT uint3 id : SV_DispatchThreadID) {
  if (id.x < SP_BINDING_build_parameters.leaf_count)
    WriteLeaf(SP_CONTEXT_ARG id.x, SP_BINDING_keys.Load(id.x * 8 + 4));
}
