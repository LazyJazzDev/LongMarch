#include "grassland/graphics/graphics_util.h"

#include "grassland/graphics/acceleration_structure.h"
#include "grassland/graphics/buffer.h"
#include "grassland/graphics/image.h"
namespace grassland::graphics {

const char *BackendAPIString(BackendAPI api) {
  switch (api) {
    case BACKEND_API_VULKAN:
      return "Vulkan";
    case BACKEND_API_D3D12:
      return "D3D12";
    default:
      return "Unknown";
  }
}

bool SupportBackendAPI(BackendAPI api) {
  switch (api) {
#if defined(LONGMARCH_D3D12_ENABLED)
    case BACKEND_API_D3D12:
      return true;
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    case BACKEND_API_VULKAN:
      return true;
#endif
    default:
      return false;
  }
}

SamplerInfo::SamplerInfo()
    : min_filter(FILTER_MODE_LINEAR),
      mag_filter(FILTER_MODE_LINEAR),
      mip_filter(FILTER_MODE_LINEAR),
      address_mode_u(ADDRESS_MODE_REPEAT),
      address_mode_v(ADDRESS_MODE_REPEAT),
      address_mode_w(ADDRESS_MODE_REPEAT) {
}

SamplerInfo::SamplerInfo(FilterMode filter) : SamplerInfo(filter, ADDRESS_MODE_REPEAT) {
}

SamplerInfo::SamplerInfo(AddressMode address_mode) : SamplerInfo(FILTER_MODE_LINEAR, address_mode) {
}

SamplerInfo::SamplerInfo(FilterMode filter, AddressMode address_mode)
    : SamplerInfo(filter, filter, filter, address_mode, address_mode, address_mode) {
}

SamplerInfo::SamplerInfo(FilterMode min_filter,
                         FilterMode mag_filter,
                         FilterMode mip_filter,
                         AddressMode address_mode_u,
                         AddressMode address_mode_v,
                         AddressMode address_mode_w)
    : min_filter(min_filter),
      mag_filter(mag_filter),
      mip_filter(mip_filter),
      address_mode_u(address_mode_u),
      address_mode_v(address_mode_v),
      address_mode_w(address_mode_w) {
}

BlendState::BlendState() {
  blend_enable = false;
  src_color = BLEND_FACTOR_ONE;
  dst_color = BLEND_FACTOR_ZERO;
  color_op = BLEND_OP_ADD;
  src_alpha = BLEND_FACTOR_ONE;
  dst_alpha = BLEND_FACTOR_ZERO;
  alpha_op = BLEND_OP_ADD;
}

BlendState::BlendState(bool blend_enable) : blend_enable(blend_enable) {
  src_color = BLEND_FACTOR_SRC_ALPHA;
  dst_color = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_op = BLEND_OP_ADD;
  src_alpha = BLEND_FACTOR_ONE;
  dst_alpha = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  alpha_op = BLEND_OP_ADD;
}

BlendState::BlendState(BlendFactor src_color,
                       BlendFactor dst_color,
                       BlendOp color_op,
                       BlendFactor src_alpha,
                       BlendFactor dst_alpha,
                       BlendOp alpha_op)
    : blend_enable(true),
      src_color(src_color),
      dst_color(dst_color),
      color_op(color_op),
      src_alpha(src_alpha),
      dst_alpha(dst_alpha),
      alpha_op(alpha_op) {
}

bool IsDepthFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_D32_SFLOAT:
      return true;
    default:
      return false;
  }
}

glm::vec3 HSVtoRGB(const glm::vec3 hsv) {
  float c = hsv.z * hsv.y;
  float h = hsv.x * 360.0 / 60.0f;
  float x = c * (1.0f - std::abs(std::fmod(h, 2.0f) - 1.0f));
  float m = hsv.z - c;
  glm::vec3 rgb;
  if (h >= 0.0f && h < 1.0f) {
    rgb = glm::vec3(c, x, 0.0f);
  } else if (h >= 1.0f && h < 2.0f) {
    rgb = glm::vec3(x, c, 0.0f);
  } else if (h >= 2.0f && h < 3.0f) {
    rgb = glm::vec3(0.0f, c, x);
  } else if (h >= 3.0f && h < 4.0f) {
    rgb = glm::vec3(0.0f, x, c);
  } else if (h >= 4.0f && h < 5.0f) {
    rgb = glm::vec3(x, 0.0f, c);
  } else if (h >= 5.0f && h < 6.0f) {
    rgb = glm::vec3(c, 0.0f, x);
  }
  return rgb + glm::vec3(m);
}

float GreyScale(const glm::vec3 rgb) {
  return 0.299f * rgb.r + 0.587f * rgb.g + 0.114f * rgb.b;
}

uint32_t PixelSize(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return 4;
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return 4;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return 12;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return 8;
    case IMAGE_FORMAT_R32_SFLOAT:
      return 4;
    case IMAGE_FORMAT_D32_SFLOAT:
      return 4;
    default:
      return 0;
  }
}

void util::PybindModuleRegistration(py::module_ &m) {
  py::enum_<BackendAPI> backend_api(m, "BackendAPI");
  backend_api.value("BACKEND_API_VULKAN", BACKEND_API_VULKAN, "Backend API: Vulkan");
  backend_api.value("BACKEND_API_D3D12", BACKEND_API_D3D12, "Backend API: Direct3D 12");
  backend_api.export_values();
  m.attr("BACKEND_API_DEFAULT") = py::cast(BACKEND_API_DEFAULT);

  py::enum_<ImageFormat> image_format(m, "ImageFormat");
  image_format.value("IMAGE_FORMAT_UNDEFINED", IMAGE_FORMAT_UNDEFINED, "Image Format: Undefined");
  image_format.value("IMAGE_FORMAT_B8G8R8A8_UNORM", IMAGE_FORMAT_B8G8R8A8_UNORM, "Image Format: B8G8R8A8 Unorm");
  image_format.value("IMAGE_FORMAT_R8G8B8A8_UNORM", IMAGE_FORMAT_R8G8B8A8_UNORM, "Image Format: R8G8B8A8 Unorm");
  image_format.value("IMAGE_FORMAT_R32G32B32A32_SFLOAT", IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                     "Image Format: R32G32B32A32 Float");
  image_format.value("IMAGE_FORMAT_R32G32B32_SFLOAT", IMAGE_FORMAT_R32G32B32_SFLOAT, "Image Format: R32G32B32 Float");
  image_format.value("IMAGE_FORMAT_R32G32_SFLOAT", IMAGE_FORMAT_R32G32_SFLOAT, "Image Format: R32G32 Float");
  image_format.value("IMAGE_FORMAT_R32_SFLOAT", IMAGE_FORMAT_R32_SFLOAT, "Image Format: R32 Float");
  image_format.value("IMAGE_FORMAT_D32_SFLOAT", IMAGE_FORMAT_D32_SFLOAT, "Image Format: D32 Float");
  image_format.value("IMAGE_FORMAT_R16G16B16A16_SFLOAT", IMAGE_FORMAT_R16G16B16A16_SFLOAT,
                     "Image Format: R16G16B16A16 Float");
  image_format.value("IMAGE_FORMAT_R32_UINT", IMAGE_FORMAT_R32_UINT, "Image Format: R32 Uint");
  image_format.value("IMAGE_FORMAT_R32_SINT", IMAGE_FORMAT_R32_SINT, "Image Format: R32 Sint");
  image_format.export_values();

  py::enum_<InputType> input_type(m, "InputType");
  input_type.value("INPUT_TYPE_UINT", INPUT_TYPE_UINT, "Input Type: uint");
  input_type.value("INPUT_TYPE_INT", INPUT_TYPE_INT, "Input Type: int");
  input_type.value("INPUT_TYPE_FLOAT", INPUT_TYPE_FLOAT, "Input Type: float");
  input_type.value("INPUT_TYPE_UINT2", INPUT_TYPE_UINT2, "Input Type: uint2");
  input_type.value("INPUT_TYPE_INT2", INPUT_TYPE_INT2, "Input Type: int2");
  input_type.value("INPUT_TYPE_FLOAT2", INPUT_TYPE_FLOAT2, "Input Type: float2");
  input_type.value("INPUT_TYPE_UINT3", INPUT_TYPE_UINT3, "Input Type: uint3");
  input_type.value("INPUT_TYPE_INT3", INPUT_TYPE_INT3, "Input Type: int3");
  input_type.value("INPUT_TYPE_FLOAT3", INPUT_TYPE_FLOAT3, "Input Type: float3");
  input_type.value("INPUT_TYPE_UINT4", INPUT_TYPE_UINT4, "Input Type: uint4");
  input_type.value("INPUT_TYPE_INT4", INPUT_TYPE_INT4, "Input Type: int4");
  input_type.value("INPUT_TYPE_FLOAT4", INPUT_TYPE_FLOAT4, "Input Type: float4");
  input_type.export_values();

  py::enum_<BufferType> buffer_type(m, "BufferType");
  buffer_type.value("BUFFER_TYPE_STATIC", BUFFER_TYPE_STATIC, "Buffer Type: Static");
  buffer_type.value("BUFFER_TYPE_DYNAMIC", BUFFER_TYPE_DYNAMIC, "Buffer Type: Dynamic");
  buffer_type.value("BUFFER_TYPE_ONETIME", BUFFER_TYPE_ONETIME, "Buffer Type: One-Time");
  buffer_type.export_values();

  py::enum_<ResourceType> resource_type(m, "ResourceType");
  resource_type.value("RESOURCE_TYPE_UNIFORM_BUFFER", RESOURCE_TYPE_UNIFORM_BUFFER, "Resource Type: Uniform Buffer");
  resource_type.value("RESOURCE_TYPE_STORAGE_BUFFER", RESOURCE_TYPE_STORAGE_BUFFER, "Resource Type: Storage Buffer");
  resource_type.value("RESOURCE_TYPE_IMAGE", RESOURCE_TYPE_IMAGE, "Resource Type: Image (read-only)");
  resource_type.value("RESOURCE_TYPE_WRITABLE_IMAGE", RESOURCE_TYPE_WRITABLE_IMAGE, "Resource Type: Writable Image");
  resource_type.value("RESOURCE_TYPE_SAMPLER", RESOURCE_TYPE_SAMPLER, "Resource Type: Sampler");
  resource_type.value("RESOURCE_TYPE_ACCELERATION_STRUCTURE", RESOURCE_TYPE_ACCELERATION_STRUCTURE,
                      "Resource Type: Acceleration Structure");
  resource_type.value("RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER", RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER,
                      "Resource Type: Writable Storage Buffer");
  resource_type.export_values();

  py::enum_<ShaderType> shader_type(m, "ShaderType");
  shader_type.value("SHADER_TYPE_VERTEX", SHADER_TYPE_VERTEX, "Shader Type: Vertex Shader");
  shader_type.value("SHADER_TYPE_PIXEL", SHADER_TYPE_PIXEL, "Shader Type: Pixel Shader");
  shader_type.value("SHADER_TYPE_GEOMETRY", SHADER_TYPE_GEOMETRY, "Shader Type: Geometry Shader");
  shader_type.export_values();
  m.attr("SHADER_TYPE_FRAGMENT") = py::cast(SHADER_TYPE_PIXEL);

  py::enum_<BindPoint> bind_point(m, "BindPoint");
  bind_point.value("BIND_POINT_GRAPHICS", BIND_POINT_GRAPHICS, "Bind Point: Graphics Pipeline");
  bind_point.value("BIND_POINT_COMPUTE", BIND_POINT_COMPUTE, "Bind Point: Compute Pipeline");
  bind_point.value("BIND_POINT_RAYTRACING", BIND_POINT_RAYTRACING, "Bind Point: Ray Tracing Pipeline");
  bind_point.export_values();

  py::enum_<CullMode> cull_mode(m, "CullMode");
  cull_mode.value("CULL_MODE_NONE", CULL_MODE_NONE, "Cull Mode: None");
  cull_mode.value("CULL_MODE_FRONT", CULL_MODE_FRONT, "Cull Mode: Front Face");
  cull_mode.value("CULL_MODE_BACK", CULL_MODE_BACK, "Cull Mode: Back Face");
  cull_mode.export_values();

  py::enum_<FilterMode> filter_mode(m, "FilterMode");
  filter_mode.value("FILTER_MODE_NEAREST", FILTER_MODE_NEAREST, "Filter Mode: Nearest");
  filter_mode.value("FILTER_MODE_LINEAR", FILTER_MODE_LINEAR, "Filter Mode: Linear");
  filter_mode.export_values();

  py::enum_<AddressMode> address_mode(m, "AddressMode");
  address_mode.value("ADDRESS_MODE_REPEAT", ADDRESS_MODE_REPEAT, "Address Mode: Repeat");
  address_mode.value("ADDRESS_MODE_MIRRORED_REPEAT", ADDRESS_MODE_MIRRORED_REPEAT, "Address Mode: Mirrored Repeat");
  address_mode.value("ADDRESS_MODE_CLAMP_TO_EDGE", ADDRESS_MODE_CLAMP_TO_EDGE, "Address Mode: Clamp to Edge");
  address_mode.value("ADDRESS_MODE_CLAMP_TO_BORDER", ADDRESS_MODE_CLAMP_TO_BORDER, "Address Mode: Clamp to Border");
  address_mode.export_values();

  py::class_<SamplerInfo> sampler_info(m, "SamplerInfo");

  sampler_info.def(py::init([](FilterMode filter_mode) {
                     SamplerInfo info{filter_mode};
                     return info;
                   }),
                   py::arg_v("filter", FILTER_MODE_LINEAR, "FilterMode()"), "Create a sampler info");
  sampler_info.def(py::init([](AddressMode address_mode) {
                     SamplerInfo info{address_mode};
                     return info;
                   }),
                   py::arg_v("address_mode", ADDRESS_MODE_REPEAT, "AddressMode()"), "Create a sampler info");
  sampler_info.def(py::init([](FilterMode filter_mode, AddressMode address_mode) {
                     SamplerInfo info{filter_mode, address_mode};
                     return info;
                   }),
                   py::arg_v("filter", FILTER_MODE_LINEAR, "FilterMode()"),
                   py::arg_v("address_mode", ADDRESS_MODE_REPEAT, "AddressMode()"), "Create a sampler info");
  sampler_info.def(py::init([](FilterMode min_filter, FilterMode mag_filter, FilterMode mip_filter,
                               AddressMode address_mode_u, AddressMode address_mode_v, AddressMode address_mode_w) {
                     SamplerInfo info{min_filter,     mag_filter,     mip_filter,
                                      address_mode_u, address_mode_v, address_mode_w};
                     return info;
                   }),
                   py::arg_v("min_filter", FILTER_MODE_LINEAR, "FilterMode()"),
                   py::arg_v("mag_filter", FILTER_MODE_LINEAR, "FilterMode()"),
                   py::arg_v("mip_filter", FILTER_MODE_LINEAR, "FilterMode()"),
                   py::arg_v("address_mode_u", ADDRESS_MODE_REPEAT, "AddressMode()"),
                   py::arg_v("address_mode_v", ADDRESS_MODE_REPEAT, "AddressMode()"),
                   py::arg_v("address_mode_w", ADDRESS_MODE_REPEAT, "AddressMode()"), "Create a sampler info");
  sampler_info.def_readwrite("min_filter", &SamplerInfo::min_filter, "Minification filter mode");
  sampler_info.def_readwrite("mag_filter", &SamplerInfo::mag_filter, "Magnification filter mode");
  sampler_info.def_readwrite("mip_filter", &SamplerInfo::mip_filter, "Mipmap filter mode");
  sampler_info.def_readwrite("address_mode_u", &SamplerInfo::address_mode_u, "Address mode for U coordinate");
  sampler_info.def_readwrite("address_mode_v", &SamplerInfo::address_mode_v, "Address mode for V coordinate");
  sampler_info.def_readwrite("address_mode_w", &SamplerInfo::address_mode_w, "Address mode for W coordinate");
  sampler_info.def("__repr__", [](const SamplerInfo &info) {
    return py::str(
               "SamplerInfo(min_filter={}, mag_filter={}, mip_filter={}, address_mode_u={}, address_mode_v={}, "
               "address_mode_w={})")
        .format(info.min_filter, info.mag_filter, info.mip_filter, info.address_mode_u, info.address_mode_v,
                info.address_mode_w);
  });

  py::enum_<PrimitiveTopology> primitive_topology(m, "PrimitiveTopology");
  primitive_topology.value("PRIMITIVE_TOPOLOGY_TRIANGLE_LIST", PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                           "Primitive Topology: Triangle List");
  primitive_topology.value("PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP", PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                           "Primitive Topology: Triangle Strip");
  primitive_topology.value("PRIMITIVE_TOPOLOGY_LINE_LIST", PRIMITIVE_TOPOLOGY_LINE_LIST,
                           "Primitive Topology: Line List");
  primitive_topology.value("PRIMITIVE_TOPOLOGY_LINE_STRIP", PRIMITIVE_TOPOLOGY_LINE_STRIP,
                           "Primitive Topology: Line Strip");
  primitive_topology.value("PRIMITIVE_TOPOLOGY_POINT_LIST", PRIMITIVE_TOPOLOGY_POINT_LIST,
                           "Primitive Topology: Point List");
  primitive_topology.export_values();

  py::enum_<BlendFactor> blend_factor(m, "BlendFactor");
  blend_factor.value("BLEND_FACTOR_ZERO", BLEND_FACTOR_ZERO, "Blend Factor: Zero");
  blend_factor.value("BLEND_FACTOR_ONE", BLEND_FACTOR_ONE, "Blend Factor: One");
  blend_factor.value("BLEND_FACTOR_SRC_COLOR", BLEND_FACTOR_SRC_COLOR, "Blend Factor: Source Color");
  blend_factor.value("BLEND_FACTOR_ONE_MINUS_SRC_COLOR", BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                     "Blend Factor: One Minus Source Color");
  blend_factor.value("BLEND_FACTOR_DST_COLOR", BLEND_FACTOR_DST_COLOR, "Blend Factor: Destination Color");
  blend_factor.value("BLEND_FACTOR_ONE_MINUS_DST_COLOR", BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                     "Blend Factor: One Minus Destination Color");
  blend_factor.value("BLEND_FACTOR_SRC_ALPHA", BLEND_FACTOR_SRC_ALPHA, "Blend Factor: Source Alpha");
  blend_factor.value("BLEND_FACTOR_ONE_MINUS_SRC_ALPHA", BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                     "Blend Factor: One Minus Source Alpha");
  blend_factor.value("BLEND_FACTOR_DST_ALPHA", BLEND_FACTOR_DST_ALPHA, "Blend Factor: Destination Alpha");
  blend_factor.value("BLEND_FACTOR_ONE_MINUS_DST_ALPHA", BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                     "Blend Factor: One Minus Destination Alpha");
  blend_factor.export_values();

  py::enum_<BlendOp> blend_op(m, "BlendOp");
  blend_op.value("BLEND_OP_ADD", BLEND_OP_ADD, "Blend Operation: Add");
  blend_op.value("BLEND_OP_SUBTRACT", BLEND_OP_SUBTRACT, "Blend Operation: Subtract");
  blend_op.value("BLEND_OP_REVERSE_SUBTRACT", BLEND_OP_REVERSE_SUBTRACT, "Blend Operation: Reverse Subtract");
  blend_op.value("BLEND_OP_MIN", BLEND_OP_MIN, "Blend Operation: Min");
  blend_op.value("BLEND_OP_MAX", BLEND_OP_MAX, "Blend Operation: Max");
  blend_op.export_values();

  py::class_<BlendState> blend_state(m, "BlendState");
  blend_state.def(py::init<>(), "Create a default blend state (no blending)");
  blend_state.def(py::init([](bool blend_enable) {
                    BlendState state{blend_enable};
                    return state;
                  }),
                  py::arg("blend_enable"), "Create a default blend state");
  blend_state.def(py::init([](BlendFactor src_color, BlendFactor dst_color, BlendOp color_op, BlendFactor src_alpha,
                              BlendFactor dst_alpha, BlendOp alpha_op) {
                    BlendState state{src_color, dst_color, color_op, src_alpha, dst_alpha, alpha_op};
                    return state;
                  }),
                  py::arg_v("src_color", BLEND_FACTOR_SRC_ALPHA, "src_color"),
                  py::arg_v("dst_color", BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, "dst_color"),
                  py::arg_v("color_op", BLEND_OP_ADD, "color_op"),
                  py::arg_v("src_alpha", BLEND_FACTOR_ONE, "src_alpha"),
                  py::arg_v("dst_alpha", BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, "dst_alpha"),
                  py::arg_v("alpha_op", BLEND_OP_ADD, "alpha_op"), "Create a custom blend state");
  blend_state.def("__repr__", [](const BlendState &state) {
    return py::str(
               "BlendState(blend_enable={}, src_color={}, dst_color={}, color_op={}, src_alpha={}, dst_alpha={}, "
               "alpha_op={})")
        .format(state.blend_enable, state.src_color, state.dst_color, state.color_op, state.src_alpha, state.dst_alpha,
                state.alpha_op);
  });
  blend_state.def_readwrite("blend_enable", &BlendState::blend_enable, "Enable or disable blending");
  blend_state.def_readwrite("src_color", &BlendState::src_color, "Source color blend factor");
  blend_state.def_readwrite("dst_color", &BlendState::dst_color, "Destination color blend factor");
  blend_state.def_readwrite("color_op", &BlendState::color_op, "Color blend operation");
  blend_state.def_readwrite("src_alpha", &BlendState::src_alpha, "Source alpha blend factor");
  blend_state.def_readwrite("dst_alpha", &BlendState::dst_alpha, "Destination alpha blend factor");
  blend_state.def_readwrite("alpha_op", &BlendState::alpha_op, "Alpha blend operation");

  // Ray Tracing Enums
  py::enum_<RayTracingGeometryFlag> ray_tracing_geometry_flag(m, "RayTracingGeometryFlag");
  ray_tracing_geometry_flag.value("RAYTRACING_GEOMETRY_FLAG_NONE", RAYTRACING_GEOMETRY_FLAG_NONE,
                                  "Ray Tracing Geometry Flag: None");
  ray_tracing_geometry_flag.value("RAYTRACING_GEOMETRY_FLAG_OPAQUE", RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                                  "Ray Tracing Geometry Flag: Opaque");
  ray_tracing_geometry_flag.value("RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION",
                                  RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION,
                                  "Ray Tracing Geometry Flag: No Duplicate Any Hit Invocation");
  ray_tracing_geometry_flag.export_values();

  py::enum_<RayTracingInstanceFlag> ray_tracing_instance_flag(m, "RayTracingInstanceFlag");
  ray_tracing_instance_flag.value("RAYTRACING_INSTANCE_FLAG_NONE", RAYTRACING_INSTANCE_FLAG_NONE,
                                  "Ray Tracing Instance Flag: None");
  ray_tracing_instance_flag.value("RAYTRACING_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE",
                                  RAYTRACING_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE,
                                  "Ray Tracing Instance Flag: Triangle Cull Disable");
  ray_tracing_instance_flag.value("RAYTRACING_INSTANCE_FLAG_TRIANGLE_FLIP_FACING",
                                  RAYTRACING_INSTANCE_FLAG_TRIANGLE_FLIP_FACING,
                                  "Ray Tracing Instance Flag: Triangle Front Counterclockwise");
  ray_tracing_instance_flag.value("RAYTRACING_INSTANCE_FLAG_OPAQUE", RAYTRACING_INSTANCE_FLAG_OPAQUE,
                                  "Ray Tracing Instance Flag: Force Opaque");
  ray_tracing_instance_flag.value("RAYTRACING_INSTANCE_FLAG_NO_OPAQUE", RAYTRACING_INSTANCE_FLAG_NO_OPAQUE,
                                  "Ray Tracing Instance Flag: Force Non-Opaque");
  ray_tracing_instance_flag.export_values();

  // Struct Classes
  py::class_<BufferRange> buffer_range(m, "BufferRange");
  buffer_range.def(py::init<Buffer *, uint64_t, uint64_t>(), py::arg("buffer"), py::arg("offset") = 0,
                   py::arg("size") = 0, "Create a buffer range");
  buffer_range.def_readwrite("buffer", &BufferRange::buffer, "Buffer object");
  buffer_range.def_readwrite("offset", &BufferRange::offset, "Offset in bytes");
  buffer_range.def_readwrite("size", &BufferRange::size, "Size in bytes");
  buffer_range.def("__repr__", [](const BufferRange &range) {
    return py::str("BufferRange(buffer={}, offset={}, size={})").format(range.buffer, range.offset, range.size);
  });

  py::class_<Extent2D> extent2d(m, "Extent2D");
  extent2d.def(py::init<uint32_t, uint32_t>(), py::arg("width"), py::arg("height"), "Create a 2D extent");
  extent2d.def_readwrite("width", &Extent2D::width, "Width");
  extent2d.def_readwrite("height", &Extent2D::height, "Height");
  extent2d.def("__repr__", [](const Extent2D &extent) {
    return py::str("Extent2D(width={}, height={})").format(extent.width, extent.height);
  });

  py::class_<Offset2D> offset2d(m, "Offset2D");
  offset2d.def(py::init<int, int>(), py::arg("x"), py::arg("y"), "Create a 2D offset");
  offset2d.def_readwrite("x", &Offset2D::x, "X coordinate");
  offset2d.def_readwrite("y", &Offset2D::y, "Y coordinate");
  offset2d.def("__repr__",
               [](const Offset2D &offset) { return py::str("Offset2D(x={}, y={})").format(offset.x, offset.y); });

  py::class_<Viewport> viewport(m, "Viewport");
  viewport.def(py::init<float, float, float, float, float, float>(), py::arg("x"), py::arg("y"), py::arg("width"),
               py::arg("height"), py::arg("min_depth") = 0.0f, py::arg("max_depth") = 1.0f, "Create a viewport");
  viewport.def_readwrite("x", &Viewport::x, "X coordinate");
  viewport.def_readwrite("y", &Viewport::y, "Y coordinate");
  viewport.def_readwrite("width", &Viewport::width, "Width");
  viewport.def_readwrite("height", &Viewport::height, "Height");
  viewport.def_readwrite("min_depth", &Viewport::min_depth, "Minimum depth");
  viewport.def_readwrite("max_depth", &Viewport::max_depth, "Maximum depth");
  viewport.def("__repr__", [](const Viewport &vp) {
    return py::str("Viewport(x={}, y={}, width={}, height={}, min_depth={}, max_depth={})")
        .format(vp.x, vp.y, vp.width, vp.height, vp.min_depth, vp.max_depth);
  });

  py::class_<Scissor> scissor(m, "Scissor");
  scissor.def(py::init<Offset2D, Extent2D>(), py::arg("offset"), py::arg("extent"), "Create a scissor");
  scissor.def_readwrite("offset", &Scissor::offset, "Offset");
  scissor.def_readwrite("extent", &Scissor::extent, "Extent");
  scissor.def("__repr__",
              [](const Scissor &s) { return py::str("Scissor(offset={}, extent={})").format(s.offset, s.extent); });

  py::class_<RayTracingInstance> ray_tracing_instance(m, "RayTracingInstance");
  ray_tracing_instance.def(py::init<>(), "Create a ray tracing instance");

  // Transform matrix property (3x4 array)
  ray_tracing_instance.def_property(
      "transform",
      [](const RayTracingInstance &instance) {
        py::list result;
        for (int i = 0; i < 3; i++) {
          py::list row;
          for (int j = 0; j < 4; j++) {
            row.append(instance.transform[i][j]);
          }
          result.append(row);
        }
        return result;
      },
      [](RayTracingInstance &instance, py::list transform_list) {
        if (transform_list.size() != 3) {
          throw std::runtime_error("Transform matrix must have 3 rows");
        }
        for (int i = 0; i < 3; i++) {
          py::list row = transform_list[i].cast<py::list>();
          if (row.size() != 4) {
            throw std::runtime_error("Transform matrix rows must have 4 columns");
          }
          for (int j = 0; j < 4; j++) {
            instance.transform[i][j] = row[j].cast<float>();
          }
        }
      },
      "Transform matrix (3x4)");

  // Bit field properties
  ray_tracing_instance.def_property(
      "instance_id", [](const RayTracingInstance &instance) { return instance.instance_id; },
      [](RayTracingInstance &instance, uint32_t value) { instance.instance_id = value; }, "Instance ID (24-bit)");

  ray_tracing_instance.def_property(
      "instance_mask", [](const RayTracingInstance &instance) { return instance.instance_mask; },
      [](RayTracingInstance &instance, uint32_t value) { instance.instance_mask = value; }, "Instance mask (8-bit)");

  ray_tracing_instance.def_property(
      "instance_hit_group_offset",
      [](const RayTracingInstance &instance) { return instance.instance_hit_group_offset; },
      [](RayTracingInstance &instance, uint32_t value) { instance.instance_hit_group_offset = value; },
      "Instance hit group offset (24-bit)");

  ray_tracing_instance.def_property(
      "instance_flags", [](const RayTracingInstance &instance) { return instance.instance_flags; },
      [](RayTracingInstance &instance, RayTracingInstanceFlag value) { instance.instance_flags = value; },
      "Instance flags (8-bit)");

  ray_tracing_instance.def_readwrite("acceleration_structure", &RayTracingInstance::acceleration_structure,
                                     "Acceleration structure");

  ray_tracing_instance.def("__repr__", [](const RayTracingInstance &instance) {
    return py::str(
               "RayTracingInstance(instance_id={}, instance_mask={}, instance_hit_group_offset={}, instance_flags={})")
        .format(instance.instance_id, instance.instance_mask, instance.instance_hit_group_offset,
                instance.instance_flags);
  });

  py::class_<RayTracingAABB> ray_tracing_aabb(m, "RayTracingAABB");
  ray_tracing_aabb.def(py::init<float, float, float, float, float, float>(), py::arg("min_x"), py::arg("min_y"),
                       py::arg("min_z"), py::arg("max_x"), py::arg("max_y"), py::arg("max_z"),
                       "Create a ray tracing AABB");
  ray_tracing_aabb.def_readwrite("min_x", &RayTracingAABB::min_x, "Minimum X coordinate");
  ray_tracing_aabb.def_readwrite("min_y", &RayTracingAABB::min_y, "Minimum Y coordinate");
  ray_tracing_aabb.def_readwrite("min_z", &RayTracingAABB::min_z, "Minimum Z coordinate");
  ray_tracing_aabb.def_readwrite("max_x", &RayTracingAABB::max_x, "Maximum X coordinate");
  ray_tracing_aabb.def_readwrite("max_y", &RayTracingAABB::max_y, "Maximum Y coordinate");
  ray_tracing_aabb.def_readwrite("max_z", &RayTracingAABB::max_z, "Maximum Z coordinate");
  ray_tracing_aabb.def("__repr__", [](const RayTracingAABB &aabb) {
    return py::str("RayTracingAABB(min=({}, {}, {}), max=({}, {}, {}))")
        .format(aabb.min_x, aabb.min_y, aabb.min_z, aabb.max_x, aabb.max_y, aabb.max_z);
  });
}

}  // namespace grassland::graphics
