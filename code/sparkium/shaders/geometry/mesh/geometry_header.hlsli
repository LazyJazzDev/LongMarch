#ifndef SPARKIUM_SHADER_GEOMETRY_MESH_GEOMETRY_HEADER_HLSLI_
#define SPARKIUM_SHADER_GEOMETRY_MESH_GEOMETRY_HEADER_HLSLI_
struct GeometryHeader {
  uint num_vertices;
  uint num_indices;
  uint position_offset;
  uint position_stride;
  uint normal_offset;
  uint normal_stride;
  uint tex_coord_offset;
  uint tex_coord_stride;
  uint tangent_offset;
  uint tangent_stride;
  uint signal_offset;
  uint signal_stride;
  uint index_offset;
  uint color_offset;
  uint color_stride;
};

#endif  // SPARKIUM_SHADER_GEOMETRY_MESH_GEOMETRY_HEADER_HLSLI_