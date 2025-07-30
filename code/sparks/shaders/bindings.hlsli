#pragma once
#include "common.hlsli"
#include "buffer_helper.hlsli"

RWTexture2D<float4> accumulated_color : register(u0, space0);
RWTexture2D<int> accumulated_samples : register(u0, space1);

RaytracingAccelerationStructure as : register(t0, space2);

ConstantBuffer<SceneSettings> scene_settings : register(b0, space3);

ByteAddressBuffer camera_data : register(t0, space4);
ByteAddressBuffer geometry_data[] : register(t0, space5);
ByteAddressBuffer surface_data[] : register(t0, space6);
StructuredBuffer<SurfaceRegistration> surface_regs : register(t0, space7);

ByteAddressBuffer light_sampler_data : register(t0, space8);
ByteAddressBuffer direct_lighting_sampler_data[] : register(t0, space9);
// StructuredBuffer<>
