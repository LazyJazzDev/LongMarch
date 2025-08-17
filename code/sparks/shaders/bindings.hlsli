#pragma once
#include "common.hlsli"
#include "buffer_helper.hlsli"

RWTexture2D<float4> accumulated_color : register(u0, space0);
RWTexture2D<float> accumulated_samples : register(u0, space1);

RaytracingAccelerationStructure as : register(t0, space2);

ConstantBuffer<RenderSettings> render_settings : register(b0, space3);

ByteAddressBuffer camera_data : register(t0, space4);
ByteAddressBuffer data_buffers[] : register(t0, space5);
StructuredBuffer<InstanceMetadata> instance_metadatas : register(t0, space6);

ByteAddressBuffer light_selector_data : register(t0, space7);
StructuredBuffer<LightMetadata> light_metadatas : register(t0, space8);
