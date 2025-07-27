
struct MaterialRegistration {
  int shader_index;
  int buffer_index;
};

RWTexture2D<float4> accumulated_color : register(u0, space0);
RWTexture2D<int> accumulated_samples : register(u0, space1);

RaytracingAccelerationStructure as : register(t0, space2);

ByteAddressBuffer camera_data : register(t0, space3);
ByteAddressBuffer geometry_data[] : register(t0, space4);
ByteAddressBuffer material_data[] : register(t0, space5);
StructuredBuffer<MaterialRegistration> material_regs : register(t0, space6);

float4 LoadFloat4(ByteAddressBuffer buf, uint baseOffset) {
    uint4 vals = buf.Load4(baseOffset); // 16 bytes per float4
    return asfloat(vals);
}

float3 LoadFloat3(ByteAddressBuffer buf, uint baseOffset) {
    uint3 vals = buf.Load3(baseOffset); // 16 bytes per float3
    return asfloat(vals);
}

float2 LoadFloat2(ByteAddressBuffer buf, uint baseOffset) {
    uint2 vals = buf.Load2(baseOffset); // 16 bytes per float2
    return asfloat(vals);
}

float LoadFloat(ByteAddressBuffer buf, uint baseOffset) {
    return asfloat(buf.Load(baseOffset)); // 4 bytes per float
}

float4x4 LoadFloat4x4(ByteAddressBuffer buf, uint baseOffset) {
    float4x4 mat;
    mat[0] = LoadFloat4(buf, baseOffset + 0);
    mat[1] = LoadFloat4(buf, baseOffset + 16);
    mat[2] = LoadFloat4(buf, baseOffset + 32);
    mat[3] = LoadFloat4(buf, baseOffset + 48);
    return transpose(mat); // Transpose for row-major order
}

float4x3 LoadFloat4x3(ByteAddressBuffer buf, uint baseOffset) {
    float3x4 mat;
    mat[0] = LoadFloat4(buf, baseOffset + 0);
    mat[1] = LoadFloat4(buf, baseOffset + 16);
    mat[2] = LoadFloat4(buf, baseOffset + 32);
    return transpose(mat); // Transpose for row-major order
}

float4x2 LoadFloat4x2(ByteAddressBuffer buf, uint baseOffset) {
    float2x4 mat;
    mat[0] = LoadFloat4(buf, baseOffset + 0);
    mat[1] = LoadFloat4(buf, baseOffset + 16);
    return transpose(mat); // Transpose for row-major order
}

float3x4 LoadFloat3x4(ByteAddressBuffer buf, uint baseOffset) {
    float4x3 mat;
    mat[0] = LoadFloat3(buf, baseOffset + 0);
    mat[1] = LoadFloat3(buf, baseOffset + 12);
    mat[2] = LoadFloat3(buf, baseOffset + 24);
    mat[3] = LoadFloat3(buf, baseOffset + 36);
    return transpose(mat); // Transpose for row-major order
}

float3x3 LoadFloat3x3(ByteAddressBuffer buf, uint baseOffset) {
    float3x3 mat;
    mat[0] = LoadFloat3(buf, baseOffset + 0);
    mat[1] = LoadFloat3(buf, baseOffset + 12);
    mat[2] = LoadFloat3(buf, baseOffset + 24);
    return transpose(mat); // Transpose for row-major order
}
