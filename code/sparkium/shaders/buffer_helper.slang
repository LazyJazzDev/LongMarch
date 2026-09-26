#pragma once

interface IBufferReader {
  uint Load(uint offset);
  uint2 Load2(uint offset);
  uint3 Load3(uint offset);
  uint4 Load4(uint offset);
};
extension ByteAddressBuffer : IBufferReader {
}
extension RWByteAddressBuffer : IBufferReader {
}

__generic<BufferType : IBufferReader> float4 LoadFloat4(BufferType buf, uint baseOffset) {
  uint4 vals = buf.Load4(baseOffset);  // 16 bytes per float4
  return asfloat(vals);
}

__generic<BufferType : IBufferReader> float3 LoadFloat3(BufferType buf, uint baseOffset) {
  uint3 vals = buf.Load3(baseOffset);  // 16 bytes per float3
  return asfloat(vals);
}

__generic<BufferType : IBufferReader> float2 LoadFloat2(BufferType buf, uint baseOffset) {
  uint2 vals = buf.Load2(baseOffset);  // 16 bytes per float2
  return asfloat(vals);
}

__generic<BufferType : IBufferReader> float LoadFloat(BufferType buf, uint baseOffset) {
  return asfloat(buf.Load(baseOffset));  // 4 bytes per float
}

__generic<BufferType : IBufferReader> float4x4 LoadFloat4x4(BufferType buf, uint baseOffset) {
  float4x4 mat;
  mat[0] = LoadFloat4(buf, baseOffset + 0);
  mat[1] = LoadFloat4(buf, baseOffset + 16);
  mat[2] = LoadFloat4(buf, baseOffset + 32);
  mat[3] = LoadFloat4(buf, baseOffset + 48);
  return transpose(mat);  // Transpose for row-major order
}

__generic<BufferType : IBufferReader> float4x3 LoadFloat4x3(BufferType buf, uint baseOffset) {
  float3x4 mat;
  mat[0] = LoadFloat4(buf, baseOffset + 0);
  mat[1] = LoadFloat4(buf, baseOffset + 16);
  mat[2] = LoadFloat4(buf, baseOffset + 32);
  return transpose(mat);  // Transpose for row-major order
}

__generic<BufferType : IBufferReader> float4x2 LoadFloat4x2(BufferType buf, uint baseOffset) {
  float2x4 mat;
  mat[0] = LoadFloat4(buf, baseOffset + 0);
  mat[1] = LoadFloat4(buf, baseOffset + 16);
  return transpose(mat);  // Transpose for row-major order
}

__generic<BufferType : IBufferReader> float3x4 LoadFloat3x4(BufferType buf, uint baseOffset) {
  float4x3 mat;
  mat[0] = LoadFloat3(buf, baseOffset + 0);
  mat[1] = LoadFloat3(buf, baseOffset + 12);
  mat[2] = LoadFloat3(buf, baseOffset + 24);
  mat[3] = LoadFloat3(buf, baseOffset + 36);
  return transpose(mat);  // Transpose for row-major order
}

__generic<BufferType : IBufferReader> float3x3 LoadFloat3x3(BufferType buf, uint baseOffset) {
  float3x3 mat;
  mat[0] = LoadFloat3(buf, baseOffset + 0);
  mat[1] = LoadFloat3(buf, baseOffset + 12);
  mat[2] = LoadFloat3(buf, baseOffset + 24);
  return transpose(mat);  // Transpose for row-major order
}

__generic<BufferType : IBufferReader> struct BufferReference : IBufferReader {
  BufferType m_buffer;
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

__generic<BufferType : IBufferReader> BufferReference<BufferType> MakeBufferReference(BufferType buffer, uint offset) {
  BufferReference<BufferType> buf;
  buf.m_buffer = buffer;
  buf.m_offset = offset;
  return buf;
}

struct RWBufferReference : IBufferReader {
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

RWBufferReference MakeBufferReference(RWByteAddressBuffer buffer, uint offset) {
  RWBufferReference result;
  result.m_buffer = buffer;
  result.m_offset = offset;
  return result;
}

__generic<BufferType : IBufferReader> struct StreamedBufferReference {
  BufferType m_buffer;
  uint m_offset;

  [mutating] float LoadFloat() {
    float result = asfloat(m_buffer.Load(m_offset));
    m_offset += 4;  // Move to the next float
    return result;
  }

      [mutating] float2 LoadFloat2() {
    float2 result = asfloat(m_buffer.Load2(m_offset));
    m_offset += 8;  // Move to the next float2
    return result;
  }

  [mutating] float3 LoadFloat3() {
    float3 result = asfloat(m_buffer.Load3(m_offset));
    m_offset += 12;  // Move to the next float3
    return result;
  }

      [mutating] float4 LoadFloat4() {
    float4 result = asfloat(m_buffer.Load4(m_offset));
    m_offset += 16;  // Move to the next float4
    return result;
  }

  [mutating] uint LoadUint() {
    uint result = m_buffer.Load(m_offset);
    m_offset += 4;  // Move to the next float
    return result;
  }

      [mutating] uint2 LoadUint2() {
    uint2 result = m_buffer.Load2(m_offset);
    m_offset += 8;  // Move to the next float
    return result;
  }

  [mutating] uint3 LoadUint3() {
    uint3 result = m_buffer.Load3(m_offset);
    m_offset += 12;  // Move to the next float
    return result;
  }

      [mutating] uint4 LoadUint4() {
    uint4 result = m_buffer.Load4(m_offset);
    m_offset += 16;  // Move to the next float
    return result;
  }

  [mutating] int LoadInt() {
    int result = m_buffer.Load(m_offset);
    m_offset += 4;  // Move to the next float
    return result;
  }

      [mutating] int2 LoadInt2() {
    int2 result = m_buffer.Load2(m_offset);
    m_offset += 8;  // Move to the next float
    return result;
  }

  [mutating] int3 LoadInt3() {
    int3 result = m_buffer.Load3(m_offset);
    m_offset += 12;  // Move to the next float
    return result;
  }

      [mutating] int4 LoadInt4() {
    int4 result = m_buffer.Load4(m_offset);
    m_offset += 16;  // Move to the next float
    return result;
  }
};

__generic<BufferType : IBufferReader> StreamedBufferReference<BufferType> MakeStreamedBufferReference(BufferType buffer,
                                                                                                      uint offset) {
  StreamedBufferReference<BufferType> buf;
  buf.m_buffer = buffer;
  buf.m_offset = offset;
  return buf;
}
