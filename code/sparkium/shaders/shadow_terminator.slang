/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0
 * Adapted from Cycles kernel/light/sample.h; see external/cycles/README.md.
 * Changes: pure Slang helpers; mesh loading/transforms handled by the caller.
 */
#pragma once

float3 ShadowTerminatorSurfaceOffset(float3 V[3], float3 N[3], float3 barycentrics, bool flipped) {
  float u = barycentrics.x, v = barycentrics.y, w = barycentrics.z;
  float3 P = V[0] * u + V[1] * v + V[2] * w;
  float3 n = N[0] * u + N[1] * v + N[2] * w;
  /* Parabolic approximation */
  const float a = dot(N[2] - N[0], V[0] - V[2]);
  const float b = dot(N[2] - N[1], V[1] - V[2]);
  const float c = dot(N[1] - N[0], V[1] - V[0]);
  float h = a * u * (u - 1) + (a + b + c) * u * v + b * v * (v - 1);

  /* Check flipped normals */
  if (!flipped) {
    /* Local linear envelope */
    float h0 = max(max(dot(V[1] - V[0], N[0]), dot(V[2] - V[0], N[0])), 0.0f);
    float h1 = max(max(dot(V[0] - V[1], N[1]), dot(V[2] - V[1], N[1])), 0.0f);
    float h2 = max(max(dot(V[0] - V[2], N[2]), dot(V[1] - V[2], N[2])), 0.0f);
    h0 = max(dot(V[0] - P, N[0]) + h0, 0.0f);
    h1 = max(dot(V[1] - P, N[1]) + h1, 0.0f);
    h2 = max(dot(V[2] - P, N[2]) + h2, 0.0f);
    h = max(min(min(h0, h1), h2), h * 0.5f);
  } else {
    float h0 = max(max(dot(V[0] - V[1], N[0]), dot(V[0] - V[2], N[0])), 0.0f);
    float h1 = max(max(dot(V[1] - V[0], N[1]), dot(V[1] - V[2], N[1])), 0.0f);
    float h2 = max(max(dot(V[2] - V[0], N[2]), dot(V[2] - V[1], N[2])), 0.0f);
    h0 = max(dot(P - V[0], N[0]) + h0, 0.0f);
    h1 = max(dot(P - V[1], N[1]) + h1, 0.0f);
    h2 = max(dot(P - V[2], N[2]) + h2, 0.0f);
    h = min(-min(min(h0, h1), h2), h * 0.5f);
  }

  return n * h;
}

// cutoff selects the grazing-angle region, not a world-space distance.
float ShadowTerminatorWeight(float NL, float NgL, float cutoff) {
  if (cutoff <= 0.0f)
    return 0.0f;
  return NL < cutoff ? saturate(2.0f - (NgL + NL) / cutoff) : saturate(1.0f - NgL / cutoff);
}
