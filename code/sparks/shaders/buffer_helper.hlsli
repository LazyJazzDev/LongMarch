#pragma once
template <class BufferType>
float4 LoadFloat4(BufferType buf, uint baseOffset) {
    uint4 vals = buf.Load4(baseOffset); // 16 bytes per float4
    return asfloat(vals);
}

template <class BufferType>
float3 LoadFloat3(BufferType buf, uint baseOffset) {
    uint3 vals = buf.Load3(baseOffset); // 16 bytes per float3
    return asfloat(vals);
}

template <class BufferType>
float2 LoadFloat2(BufferType buf, uint baseOffset) {
    uint2 vals = buf.Load2(baseOffset); // 16 bytes per float2
    return asfloat(vals);
}

template <class BufferType>
float LoadFloat(BufferType buf, uint baseOffset) {
    return asfloat(buf.Load(baseOffset)); // 4 bytes per float
}

template <class BufferType>
float4x4 LoadFloat4x4(BufferType buf, uint baseOffset) {
    float4x4 mat;
    mat[0] = LoadFloat4(buf, baseOffset + 0);
    mat[1] = LoadFloat4(buf, baseOffset + 16);
    mat[2] = LoadFloat4(buf, baseOffset + 32);
    mat[3] = LoadFloat4(buf, baseOffset + 48);
    return transpose(mat); // Transpose for row-major order
}

template <class BufferType>
float4x3 LoadFloat4x3(BufferType buf, uint baseOffset) {
    float3x4 mat;
    mat[0] = LoadFloat4(buf, baseOffset + 0);
    mat[1] = LoadFloat4(buf, baseOffset + 16);
    mat[2] = LoadFloat4(buf, baseOffset + 32);
    return transpose(mat); // Transpose for row-major order
}

template <class BufferType>
float4x2 LoadFloat4x2(BufferType buf, uint baseOffset) {
    float2x4 mat;
    mat[0] = LoadFloat4(buf, baseOffset + 0);
    mat[1] = LoadFloat4(buf, baseOffset + 16);
    return transpose(mat); // Transpose for row-major order
}

template <class BufferType>
float3x4 LoadFloat3x4(BufferType buf, uint baseOffset) {
    float4x3 mat;
    mat[0] = LoadFloat3(buf, baseOffset + 0);
    mat[1] = LoadFloat3(buf, baseOffset + 12);
    mat[2] = LoadFloat3(buf, baseOffset + 24);
    mat[3] = LoadFloat3(buf, baseOffset + 36);
    return transpose(mat); // Transpose for row-major order
}

template <class BufferType>
float3x3 LoadFloat3x3(BufferType buf, uint baseOffset) {
    float3x3 mat;
    mat[0] = LoadFloat3(buf, baseOffset + 0);
    mat[1] = LoadFloat3(buf, baseOffset + 12);
    mat[2] = LoadFloat3(buf, baseOffset + 24);
    return transpose(mat); // Transpose for row-major order
}


template <class BufferType>
class BufferReference {
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

  template <class T>
  T Load(uint offset) {
    return m_buffer.template Load<T>(m_offset + offset);
  }
};

template <class BufferType>
BufferReference<BufferType> MakeBufferReference(BufferType buffer, uint offset) {
    BufferReference<BufferType> buf;
    buf.m_buffer = buffer;
    buf.m_offset = offset;
    return buf;
}

template <>
class BufferReference<RWByteAddressBuffer> {
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

  template <class T>
  T Load(uint offset) {
    return m_buffer.template Load<T>(m_offset + offset);
  }

  template <class T>
  void Store(uint offset, T value) {
    return m_buffer.template Store<T>(m_offset + offset, value);
  }

};


template <class BufferType>
class StreamedBufferReference {
  BufferType m_buffer;
  uint m_offset;

  float LoadFloat() {
    float result = asfloat(m_buffer.Load(m_offset));
    m_offset += 4; // Move to the next float
        return result;
  }

float2 LoadFloat2() {
        float2 result = asfloat(m_buffer.Load2(m_offset));
        m_offset += 8; // Move to the next float2
        return result;
        }

float3 LoadFloat3() {
        float3 result = asfloat(m_buffer.Load3(m_offset));
        m_offset += 12; // Move to the next float3
        return result;
        }

float4 LoadFloat4() {
        float4 result = asfloat(m_buffer.Load4(m_offset));
        m_offset += 16; // Move to the next float4
        return result;
        }

};

template <class BufferType>
StreamedBufferReference<BufferType> MakeStreamedBufferReference(BufferType buffer, uint offset) {
    StreamedBufferReference<BufferType> buf;
        buf.m_buffer = buffer;
        buf.m_offset = offset;
        return buf;
}
