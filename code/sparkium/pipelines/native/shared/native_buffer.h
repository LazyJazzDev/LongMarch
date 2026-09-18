#pragma once

// Byte-address buffer accessors mirroring `shaders/buffer_helper.hlsli`. The
// backends flatten the scene into the very same byte layouts the graphics
// pipelines upload, so the ported shading core reads identical offsets.

#include "native_math.h"

namespace sparkium::native {

// Read-only view over a byte-address buffer. `Load*` offsets are in bytes and
// must be 4-byte aligned, exactly like HLSL's ByteAddressBuffer.
struct ByteBuffer {
  const uint32_t *data;
  uint32_t size;  // In bytes.

  LM_DEVICE_FUNC uint32_t Load(uint32_t offset) const {
    return data[offset >> 2];
  }

  LM_DEVICE_FUNC uint2 Load2(uint32_t offset) const {
    return uint2{Load(offset), Load(offset + 4)};
  }

  LM_DEVICE_FUNC uint3 Load3(uint32_t offset) const {
    return uint3{Load(offset), Load(offset + 4), Load(offset + 8)};
  }

  LM_DEVICE_FUNC uint4 Load4(uint32_t offset) const {
    return uint4{Load(offset), Load(offset + 4), Load(offset + 8), Load(offset + 12)};
  }

  LM_DEVICE_FUNC bool Valid() const {
    return data != nullptr;
  }
};

// Writable counterpart, used by the accumulation buffers.
struct RWByteBuffer {
  uint32_t *data;
  uint32_t size;

  LM_DEVICE_FUNC uint32_t Load(uint32_t offset) const {
    return data[offset >> 2];
  }

  LM_DEVICE_FUNC void Store(uint32_t offset, uint32_t value) const {
    data[offset >> 2] = value;
  }

  LM_DEVICE_FUNC operator ByteBuffer() const {
    ByteBuffer buf;
    buf.data = data;
    buf.size = size;
    return buf;
  }
};

LM_DEVICE_FUNC inline float LoadFloat(const ByteBuffer &buf, uint32_t base_offset) {
  return asfloat(buf.Load(base_offset));
}

LM_DEVICE_FUNC inline float2 LoadFloat2(const ByteBuffer &buf, uint32_t base_offset) {
  return float2{LoadFloat(buf, base_offset), LoadFloat(buf, base_offset + 4)};
}

LM_DEVICE_FUNC inline float3 LoadFloat3(const ByteBuffer &buf, uint32_t base_offset) {
  return float3{LoadFloat(buf, base_offset), LoadFloat(buf, base_offset + 4), LoadFloat(buf, base_offset + 8)};
}

LM_DEVICE_FUNC inline float4 LoadFloat4(const ByteBuffer &buf, uint32_t base_offset) {
  return float4{LoadFloat(buf, base_offset), LoadFloat(buf, base_offset + 4), LoadFloat(buf, base_offset + 8),
                LoadFloat(buf, base_offset + 12)};
}

// HLSL loads four consecutive float3 rows and transposes them, which puts the
// i-th loaded float3 in column i -- precisely a glm::mat4x3 column.
LM_DEVICE_FUNC inline float3x4 LoadFloat3x4(const ByteBuffer &buf, uint32_t base_offset) {
  return float3x4{LoadFloat3(buf, base_offset + 0), LoadFloat3(buf, base_offset + 12),
                  LoadFloat3(buf, base_offset + 24), LoadFloat3(buf, base_offset + 36)};
}

LM_DEVICE_FUNC inline glm::mat3 LoadFloat3x3(const ByteBuffer &buf, uint32_t base_offset) {
  return glm::mat3{LoadFloat3(buf, base_offset + 0), LoadFloat3(buf, base_offset + 12),
                   LoadFloat3(buf, base_offset + 24)};
}

LM_DEVICE_FUNC inline glm::mat4 LoadFloat4x4(const ByteBuffer &buf, uint32_t base_offset) {
  return glm::mat4{LoadFloat4(buf, base_offset + 0), LoadFloat4(buf, base_offset + 16),
                   LoadFloat4(buf, base_offset + 32), LoadFloat4(buf, base_offset + 48)};
}

// Equivalent of `MakeBufferReference`: a buffer plus a base offset.
struct BufferReference {
  ByteBuffer m_buffer;
  uint32_t m_offset;

  LM_DEVICE_FUNC uint32_t Load(uint32_t offset) const {
    return m_buffer.Load(m_offset + offset);
  }

  LM_DEVICE_FUNC uint2 Load2(uint32_t offset) const {
    return m_buffer.Load2(m_offset + offset);
  }

  LM_DEVICE_FUNC uint3 Load3(uint32_t offset) const {
    return m_buffer.Load3(m_offset + offset);
  }

  LM_DEVICE_FUNC uint4 Load4(uint32_t offset) const {
    return m_buffer.Load4(m_offset + offset);
  }

  LM_DEVICE_FUNC bool Valid() const {
    return m_buffer.Valid();
  }

  LM_DEVICE_FUNC float LoadF(uint32_t offset) const {
    return asfloat(Load(offset));
  }

  LM_DEVICE_FUNC float2 LoadF2(uint32_t offset) const {
    return float2{LoadF(offset), LoadF(offset + 4)};
  }

  LM_DEVICE_FUNC float3 LoadF3(uint32_t offset) const {
    return float3{LoadF(offset), LoadF(offset + 4), LoadF(offset + 8)};
  }

  LM_DEVICE_FUNC float4 LoadF4(uint32_t offset) const {
    return float4{LoadF(offset), LoadF(offset + 4), LoadF(offset + 8), LoadF(offset + 12)};
  }

  LM_DEVICE_FUNC float3x4 LoadF3x4(uint32_t offset) const {
    return float3x4{LoadF3(offset + 0), LoadF3(offset + 12), LoadF3(offset + 24), LoadF3(offset + 36)};
  }
};

LM_DEVICE_FUNC inline BufferReference MakeBufferReference(const ByteBuffer &buffer, uint32_t offset) {
  BufferReference buf;
  buf.m_buffer = buffer;
  buf.m_offset = offset;
  return buf;
}

// Streaming cursor, mirroring `StreamedBufferReference`.
struct StreamedBufferReference {
  ByteBuffer m_buffer;
  uint32_t m_offset;

  LM_DEVICE_FUNC float LoadFloat() {
    const float result = asfloat(m_buffer.Load(m_offset));
    m_offset += 4;
    return result;
  }

  LM_DEVICE_FUNC float2 LoadFloat2() {
    const float2 result = native::LoadFloat2(m_buffer, m_offset);
    m_offset += 8;
    return result;
  }

  LM_DEVICE_FUNC float3 LoadFloat3() {
    const float3 result = native::LoadFloat3(m_buffer, m_offset);
    m_offset += 12;
    return result;
  }

  LM_DEVICE_FUNC float4 LoadFloat4() {
    const float4 result = native::LoadFloat4(m_buffer, m_offset);
    m_offset += 16;
    return result;
  }

  LM_DEVICE_FUNC uint32_t LoadUint() {
    const uint32_t result = m_buffer.Load(m_offset);
    m_offset += 4;
    return result;
  }

  LM_DEVICE_FUNC int32_t LoadInt() {
    const int32_t result = static_cast<int32_t>(m_buffer.Load(m_offset));
    m_offset += 4;
    return result;
  }
};

LM_DEVICE_FUNC inline StreamedBufferReference MakeStreamedBufferReference(const ByteBuffer &buffer, uint32_t offset) {
  StreamedBufferReference buf;
  buf.m_buffer = buffer;
  buf.m_offset = offset;
  return buf;
}

}  // namespace sparkium::native
