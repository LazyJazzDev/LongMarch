#pragma once
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
};
