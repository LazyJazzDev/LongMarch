#pragma once

struct RealtimeParameters {
  float4x4 view_projection;
  float4x4 previous_view_projection;
  uint4 extent;
  uint4 config;  // Frame, history valid, shading divisor, maximum history length.
  float4 camera_position;
};
