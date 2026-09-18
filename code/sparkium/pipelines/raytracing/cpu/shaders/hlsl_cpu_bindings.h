// Host-side resource bindings for the CPU backend.
//
// bindings.hlsli names a fixed set of slots. On the GPU each `register(...)`
// declaration becomes a device resource; here the same names are host objects
// the pipeline fills in before a frame, which is the CPU equivalent of binding
// a descriptor set.
//
// Include order matters: the shader's struct definitions (notably
// RenderSettings from common.hlsli) have to come first so the constant buffer
// can be declared by value, and all of it has to live in the shader namespace
// so the names resolve the way they do inside the shaders.
#pragma once

#include <vector>

#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu.h"

namespace sparkium_cpu_shaders {

#include "common.hlsli"

inline ByteAddressBuffer software_nodes;

// data_buffers carries the per-instance geometry and material payloads. The GPU
// backend recompiles its shader whenever the scene buffer count changes, which
// lets it bake the count in as SOFTWARE_DATA_BUFFER_COUNT. The CPU backend is
// compiled once, so the boundary between scene buffers and the fixed slots
// appended after them is a runtime value instead; bindings.hlsli's index macros
// refer to it.
inline uint32_t software_data_buffer_count = 0;
inline std::vector<ByteAddressBuffer> data_buffers;

inline std::vector<Texture2D<float4>> sdr_textures;
inline std::vector<Texture2D<float4>> hdr_textures;
inline std::vector<SamplerState> samplers;

inline RWTexture2D<float4> accumulated_color;
inline RWTexture2D<float> accumulated_samples;
inline ConstantBuffer<RenderSettings> render_settings;

}  // namespace sparkium_cpu_shaders
