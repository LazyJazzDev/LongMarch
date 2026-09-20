#pragma once
// Shared source dialect. GPU declarations retain their normal HLSL semantics.
#ifdef SPARKIUM_NATIVE
#define ByteAddressBuffer RWByteAddressBuffer
#define SP_CLASS struct
#define SP_BUFFER_TEMPLATE
#define SP_BUFFER_TYPE RWByteAddressBuffer
#define SP_BUFFER_ARG(T)
#define SP_GEOMETRY_TEMPLATE
#define SP_GEOMETRY_TYPE GeometrySampler
#define SP_MUTATING [mutating]
#define SP_TEMPLATE_CALL
#define SP_TEXTURE(T) NativeTexture_##T
#define SP_RW_TEXTURE(T) NativeTexture_##T
#define SP_SAMPLER NativeSamplerState
#ifdef SPARKIUM_OPTIX
#define SP_RAY RayDesc
#else
#define SP_RAY NativeRayDesc
#endif
#define SP_NONUNIFORM(i) (i)
#define SP_IMAGE_FORMAT(f)
#define SP_RESOURCE(T, name, reg, slot) [NativeBinding(slot)] T name
#define SP_ARRAY_RESOURCE(T, name, reg, slot) [NativeBinding(slot)] NativeArray<T> name
#else
#define SP_CLASS class
#define SP_BUFFER_TEMPLATE template <class BufferType>
#define SP_BUFFER_TYPE BufferType
#define SP_BUFFER_ARG(T) <T>
#define SP_GEOMETRY_TEMPLATE template <class GeometrySamplerType>
#define SP_GEOMETRY_TYPE GeometrySamplerType
#define SP_MUTATING
#define SP_TEMPLATE_CALL template
#define SP_TEXTURE(T) Texture2D<T>
#define SP_RW_TEXTURE(T) RWTexture2D<T>
#define SP_SAMPLER SamplerState
#define SP_RAY RayDesc
#define SP_NONUNIFORM(i) NonUniformResourceIndex(i)
#define SP_IMAGE_FORMAT(f) [[vk::image_format(f)]]
#define SP_RESOURCE(T, name, reg, slot) T name : register(reg, space##slot)
#define SP_ARRAY_RESOURCE(T, name, reg, slot) T name[] : register(reg, space##slot)
#endif
#ifdef SPARKIUM_CPU_FUNCTIONS
#undef SP_RESOURCE
#undef SP_ARRAY_RESOURCE
#define SP_RESOURCE(T, name, reg, slot)
#define SP_ARRAY_RESOURCE(T, name, reg, slot)
#define SP_RESOURCE_ACCESS(T, name, slot) NativeResource<T>(native_context, slot)
#define SP_ARRAY_ACCESS(T, name, slot) NativeResource<NativeArray<T>>(native_context, slot)
#define SP_CONTEXT NativeContext *native_context,
#define SP_CONTEXT_ONLY NativeContext *native_context
#define SP_CONTEXT_ARG_ONLY native_context
#define SP_CONTEXT_ARG native_context,
#define SP_NUMTHREADS(x, y, z)
#else
#define SP_RESOURCE_ACCESS(T, name, slot) name
#define SP_ARRAY_ACCESS(T, name, slot) name
#define SP_CONTEXT
#define SP_CONTEXT_ONLY
#define SP_CONTEXT_ARG_ONLY
#define SP_CONTEXT_ARG
#define SP_NUMTHREADS(x, y, z) [numthreads(x, y, z)]
#endif
