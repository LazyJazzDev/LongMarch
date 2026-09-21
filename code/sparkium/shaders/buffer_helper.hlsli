#include "compute_contract.hlsli"
#pragma once

SP_BUFFER_TEMPLATE
float4 LoadFloat4(SP_BUFFER_TYPE buf, uint baseOffset) {
  uint4 vals = buf.Load4(baseOffset);  // 16 bytes per float4
  return asfloat(vals);
}

SP_BUFFER_TEMPLATE
float3 LoadFloat3(SP_BUFFER_TYPE buf, uint baseOffset) {
  uint3 vals = buf.Load3(baseOffset);  // 16 bytes per float3
  return asfloat(vals);
}

SP_BUFFER_TEMPLATE
float2 LoadFloat2(SP_BUFFER_TYPE buf, uint baseOffset) {
  uint2 vals = buf.Load2(baseOffset);  // 16 bytes per float2
  return asfloat(vals);
}

SP_BUFFER_TEMPLATE
float LoadFloat(SP_BUFFER_TYPE buf, uint baseOffset) {
  return asfloat(buf.Load(baseOffset));  // 4 bytes per float
}

SP_BUFFER_TEMPLATE
float4x4 LoadFloat4x4(SP_BUFFER_TYPE buf, uint baseOffset) {
  float4x4 mat;
  mat[0] = LoadFloat4(buf, baseOffset + 0);
  mat[1] = LoadFloat4(buf, baseOffset + 16);
  mat[2] = LoadFloat4(buf, baseOffset + 32);
  mat[3] = LoadFloat4(buf, baseOffset + 48);
  return transpose(mat);  // Transpose for row-major order
}

SP_BUFFER_TEMPLATE
float4x3 LoadFloat4x3(SP_BUFFER_TYPE buf, uint baseOffset) {
  float3x4 mat;
  mat[0] = LoadFloat4(buf, baseOffset + 0);
  mat[1] = LoadFloat4(buf, baseOffset + 16);
  mat[2] = LoadFloat4(buf, baseOffset + 32);
  return transpose(mat);  // Transpose for row-major order
}

SP_BUFFER_TEMPLATE
float4x2 LoadFloat4x2(SP_BUFFER_TYPE buf, uint baseOffset) {
  float2x4 mat;
  mat[0] = LoadFloat4(buf, baseOffset + 0);
  mat[1] = LoadFloat4(buf, baseOffset + 16);
  return transpose(mat);  // Transpose for row-major order
}

SP_BUFFER_TEMPLATE
float3x4 LoadFloat3x4(SP_BUFFER_TYPE buf, uint baseOffset) {
  float4x3 mat;
  mat[0] = LoadFloat3(buf, baseOffset + 0);
  mat[1] = LoadFloat3(buf, baseOffset + 12);
  mat[2] = LoadFloat3(buf, baseOffset + 24);
  mat[3] = LoadFloat3(buf, baseOffset + 36);
  return transpose(mat);  // Transpose for row-major order
}

SP_BUFFER_TEMPLATE
float3x3 LoadFloat3x3(SP_BUFFER_TYPE buf, uint baseOffset) {
  float3x3 mat;
  mat[0] = LoadFloat3(buf, baseOffset + 0);
  mat[1] = LoadFloat3(buf, baseOffset + 12);
  mat[2] = LoadFloat3(buf, baseOffset + 24);
  return transpose(mat);  // Transpose for row-major order
}

#ifndef SPARKIUM_COMPUTE
SP_BUFFER_TEMPLATE
SP_CLASS BufferReference {
  SP_BUFFER_TYPE m_buffer;
  uint m_offset;

  uint Load(uint offset) {
    return m_buffer.Load(m_offset + offset);
  }

  uint2 Load2(uint offset) {
    return m_buffer.Load2(m_offset + offset);
  }

  uint3 Load3(uint offset) {
    return m_buffer.Load3(m_offset + offset);
  }

  uint4 Load4(uint offset) {
    return m_buffer.Load4(m_offset + offset);
  }
};

SP_BUFFER_TEMPLATE
BufferReference SP_BUFFER_ARG(SP_BUFFER_TYPE) MakeBufferReference(SP_BUFFER_TYPE buffer, uint offset) {
  BufferReference SP_BUFFER_ARG(SP_BUFFER_TYPE) buf;
  buf.m_buffer = buffer;
  buf.m_offset = offset;
  return buf;
}

#endif
#ifndef SPARKIUM_COMPUTE
template <>
#endif

SP_CLASS BufferReference SP_BUFFER_ARG(RWByteAddressBuffer) {
  RWByteAddressBuffer m_buffer;
  uint m_offset;

  uint Load(uint offset) {
    return m_buffer.Load(m_offset + offset);
  }

  uint2 Load2(uint offset) {
    return m_buffer.Load2(m_offset + offset);
  }

  uint3 Load3(uint offset) {
    return m_buffer.Load3(m_offset + offset);
  }

  uint4 Load4(uint offset) {
    return m_buffer.Load4(m_offset + offset);
  }

  void Store(uint offset, uint value) {
    return m_buffer.Store(m_offset + offset, value);
  }

  void Store2(uint offset, uint2 value) {
    return m_buffer.Store2(m_offset + offset, asuint(value));
  }

  void Store3(uint offset, uint3 value) {
    return m_buffer.Store3(m_offset + offset, asuint(value));
  }

  void Store4(uint offset, uint4 value) {
    return m_buffer.Store4(m_offset + offset, asuint(value));
  }
};

#ifdef SPARKIUM_COMPUTE
#pragma once

float4 LoadFloat4(BufferReference buf, uint baseOffset) {
  uint4 vals = buf.Load4(baseOffset);  // 16 bytes per float4
  return asfloat(vals);
}

float3 LoadFloat3(BufferReference buf, uint baseOffset) {
  uint3 vals = buf.Load3(baseOffset);  // 16 bytes per float3
  return asfloat(vals);
}

float2 LoadFloat2(BufferReference buf, uint baseOffset) {
  uint2 vals = buf.Load2(baseOffset);  // 16 bytes per float2
  return asfloat(vals);
}

float LoadFloat(BufferReference buf, uint baseOffset) {
  return asfloat(buf.Load(baseOffset));  // 4 bytes per float
}

float4x4 LoadFloat4x4(BufferReference buf, uint baseOffset) {
  float4x4 mat;
  mat[0] = LoadFloat4(buf, baseOffset + 0);
  mat[1] = LoadFloat4(buf, baseOffset + 16);
  mat[2] = LoadFloat4(buf, baseOffset + 32);
  mat[3] = LoadFloat4(buf, baseOffset + 48);
  return transpose(mat);  // Transpose for row-major order
}

float4x3 LoadFloat4x3(BufferReference buf, uint baseOffset) {
  float3x4 mat;
  mat[0] = LoadFloat4(buf, baseOffset + 0);
  mat[1] = LoadFloat4(buf, baseOffset + 16);
  mat[2] = LoadFloat4(buf, baseOffset + 32);
  return transpose(mat);  // Transpose for row-major order
}

float4x2 LoadFloat4x2(BufferReference buf, uint baseOffset) {
  float2x4 mat;
  mat[0] = LoadFloat4(buf, baseOffset + 0);
  mat[1] = LoadFloat4(buf, baseOffset + 16);
  return transpose(mat);  // Transpose for row-major order
}

float3x4 LoadFloat3x4(BufferReference buf, uint baseOffset) {
  float4x3 mat;
  mat[0] = LoadFloat3(buf, baseOffset + 0);
  mat[1] = LoadFloat3(buf, baseOffset + 12);
  mat[2] = LoadFloat3(buf, baseOffset + 24);
  mat[3] = LoadFloat3(buf, baseOffset + 36);
  return transpose(mat);  // Transpose for row-major order
}

float3x3 LoadFloat3x3(BufferReference buf, uint baseOffset) {
  float3x3 mat;
  mat[0] = LoadFloat3(buf, baseOffset + 0);
  mat[1] = LoadFloat3(buf, baseOffset + 12);
  mat[2] = LoadFloat3(buf, baseOffset + 24);
  return transpose(mat);  // Transpose for row-major order
}

BufferReference MakeBufferReference(RWByteAddressBuffer buffer, uint offset) {
  BufferReference buf;
  buf.m_buffer = buffer;
  buf.m_offset = offset;
  return buf;
}

#endif
SP_BUFFER_TEMPLATE
SP_CLASS StreamedBufferReference {
  SP_BUFFER_TYPE m_buffer;
  uint m_offset;

  SP_MUTATING float LoadFloat() {
    float result = asfloat(m_buffer.Load(m_offset));
    m_offset += 4;  // Move to the next float
    return result;
  }

  SP_MUTATING float2 LoadFloat2() {
    float2 result = asfloat(m_buffer.Load2(m_offset));
    m_offset += 8;  // Move to the next float2
    return result;
  }

  SP_MUTATING float3 LoadFloat3() {
    float3 result = asfloat(m_buffer.Load3(m_offset));
    m_offset += 12;  // Move to the next float3
    return result;
  }

  SP_MUTATING float4 LoadFloat4() {
    float4 result = asfloat(m_buffer.Load4(m_offset));
    m_offset += 16;  // Move to the next float4
    return result;
  }

  SP_MUTATING uint LoadUint() {
    uint result = m_buffer.Load(m_offset);
    m_offset += 4;  // Move to the next float
    return result;
  }

  SP_MUTATING uint2 LoadUint2() {
    uint2 result = m_buffer.Load2(m_offset);
    m_offset += 8;  // Move to the next float
    return result;
  }

  SP_MUTATING uint3 LoadUint3() {
    uint3 result = m_buffer.Load3(m_offset);
    m_offset += 12;  // Move to the next float
    return result;
  }

  SP_MUTATING uint4 LoadUint4() {
    uint4 result = m_buffer.Load4(m_offset);
    m_offset += 16;  // Move to the next float
    return result;
  }

  SP_MUTATING int LoadInt() {
    int result = m_buffer.Load(m_offset);
    m_offset += 4;  // Move to the next float
    return result;
  }

  SP_MUTATING int2 LoadInt2() {
    int2 result = m_buffer.Load2(m_offset);
    m_offset += 8;  // Move to the next float
    return result;
  }

  SP_MUTATING int3 LoadInt3() {
    int3 result = m_buffer.Load3(m_offset);
    m_offset += 12;  // Move to the next float
    return result;
  }

  SP_MUTATING int4 LoadInt4() {
    int4 result = m_buffer.Load4(m_offset);
    m_offset += 16;  // Move to the next float
    return result;
  }
};

SP_BUFFER_TEMPLATE
StreamedBufferReference SP_BUFFER_ARG(SP_BUFFER_TYPE) MakeStreamedBufferReference(SP_BUFFER_TYPE buffer, uint offset) {
  StreamedBufferReference SP_BUFFER_ARG(SP_BUFFER_TYPE) buf;
  buf.m_buffer = buffer;
  buf.m_offset = offset;
  return buf;
}
