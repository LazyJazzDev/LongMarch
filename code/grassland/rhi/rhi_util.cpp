#include "grassland/rhi/rhi_util.h"

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi {
uint32_t ImageFormatSize(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_R8_UINT:
      return 1;
    case IMAGE_FORMAT_R8_INT:
      return 1;
    case IMAGE_FORMAT_R8_UNORM:
      return 1;
    case IMAGE_FORMAT_R8G8_UINT:
      return 2;
    case IMAGE_FORMAT_R8G8_INT:
      return 2;
    case IMAGE_FORMAT_R8G8_UNORM:
      return 2;
    case IMAGE_FORMAT_R8G8B8_UINT:
      return 3;
    case IMAGE_FORMAT_R8G8B8_INT:
      return 3;
    case IMAGE_FORMAT_R8G8B8_UNORM:
      return 3;
    case IMAGE_FORMAT_R8G8B8A8_UINT:
      return 4;
    case IMAGE_FORMAT_R8G8B8A8_INT:
      return 4;
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return 4;
    case IMAGE_FORMAT_R32_UINT:
      return 4;
    case IMAGE_FORMAT_R32_INT:
      return 4;
    case IMAGE_FORMAT_R32_SFLOAT:
      return 4;
    case IMAGE_FORMAT_R32G32_UINT:
      return 8;
    case IMAGE_FORMAT_R32G32_INT:
      return 8;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return 8;
    case IMAGE_FORMAT_R32G32B32_UINT:
      return 12;
    case IMAGE_FORMAT_R32G32B32_INT:
      return 12;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return 12;
    case IMAGE_FORMAT_R32G32B32A32_UINT:
      return 16;
    case IMAGE_FORMAT_R32G32B32A32_INT:
      return 16;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return 4;
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
      return 8;
    case IMAGE_FORMAT_R10G10B10A2_UNORM:
      return 4;
    case IMAGE_FORMAT_B10G10R10A2_UNORM:
      return 4;
    case IMAGE_FORMAT_D16_UNORM:
      return 2;
    case IMAGE_FORMAT_D32_SFLOAT:
      return 4;
    case IMAGE_FORMAT_D24_UNORM_S8_UINT:
      return 4;
    default:
      return 0;
  }
}

bool IsColorFormat(ImageFormat format) {
  return ImageFormatSize(format) != 0 && !IsDepthFormat(format);
}

std::string to_string(ColorSpace value) {
  switch (value) {
    case COLOR_SPACE_SRGB_NONLINEAR:
      return "COLOR_SPACE_SRGB_NONLINEAR";
    case COLOR_SPACE_EXTENDED_SRGB_LINEAR:
      return "COLOR_SPACE_EXTENDED_SRGB_LINEAR";
    case COLOR_SPACE_HDR10_ST2084:
      return "COLOR_SPACE_HDR10_ST2084";
    case COLOR_SPACE_BT709_LINEAR:
      return "COLOR_SPACE_BT709_LINEAR";
    default:
      return "COLOR_SPACE_UNKNOWN";
  }
}

std::string ColorSpaceName(ColorSpace value) {
  switch (value) {
    case COLOR_SPACE_SRGB_NONLINEAR:
      return "SDR sRGB";
    case COLOR_SPACE_EXTENDED_SRGB_LINEAR:
      return "HDR linear extended sRGB";
    case COLOR_SPACE_HDR10_ST2084:
      return "HDR10 Rec.2020 PQ";
    case COLOR_SPACE_BT709_LINEAR:
      return "Linear Rec.709";
    default:
      return "Unknown color space";
  }
}

bool IsDepthFormat(ImageFormat format) {
  return format == IMAGE_FORMAT_D16_UNORM || format == IMAGE_FORMAT_D32_SFLOAT ||
         format == IMAGE_FORMAT_D24_UNORM_S8_UINT;
}

bool HasStencil(ImageFormat format) {
  return format == IMAGE_FORMAT_D24_UNORM_S8_UINT;
}

using backend::ReportError;

std::string to_string(BackendAPI api) {
  switch (api) {
    case BACKEND_API_D3D12:
      return "BACKEND_API_D3D12";
    case BACKEND_API_VULKAN:
      return "BACKEND_API_VULKAN";
    case BACKEND_API_METAL:
      return "BACKEND_API_METAL";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown API type.");
#endif
      return {};
  }
}

std::string BackendAPIName(BackendAPI api) {
  switch (api) {
    case BACKEND_API_D3D12:
      return "D3D12";
    case BACKEND_API_VULKAN:
      return "Vulkan";
    case BACKEND_API_METAL:
      return "Metal";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown API type.");
#endif
      return {};
  }
}

std::string to_string(PhysicalDeviceType type) {
  switch (type) {
    case PHYSICAL_DEVICE_TYPE_INTEGRATED:
      return "PHYSICAL_DEVICE_TYPE_INTEGRATED";
    case PHYSICAL_DEVICE_TYPE_DEDICATED:
      return "PHYSICAL_DEVICE_TYPE_DEDICATED";
    case PHYSICAL_DEVICE_TYPE_SOFTWARE:
      return "PHYSICAL_DEVICE_TYPE_SOFTWARE";
    case PHYSICAL_DEVICE_TYPE_OTHER:
      return "PHYSICAL_DEVICE_TYPE_OTHER";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown physical device type.");
#endif
      return {};
  }
}

std::string PhysicalDeviceTypeName(PhysicalDeviceType type) {
  switch (type) {
    case PHYSICAL_DEVICE_TYPE_DEDICATED:
      return "Dedicated";
    case PHYSICAL_DEVICE_TYPE_INTEGRATED:
      return "Integrated";
    case PHYSICAL_DEVICE_TYPE_SOFTWARE:
      return "Software";
    case PHYSICAL_DEVICE_TYPE_OTHER:
      return "Other";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown physical device type.");
#endif
      return {};
  }
}

std::string to_string(PresentMode value) {
  switch (value) {
    case PRESENT_MODE_VSYNC:
      return "PRESENT_MODE_VSYNC";
    case PRESENT_MODE_CONTINUOUS:
      return "PRESENT_MODE_CONTINUOUS";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown PresentMode value.");
#endif
      return {};
  }
}

std::string PresentModeName(PresentMode value) {
  switch (value) {
    case PRESENT_MODE_VSYNC:
      return "VSync";
    case PRESENT_MODE_CONTINUOUS:
      return "Continuous";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown PresentMode value.");
#endif
      return {};
  }
}

std::string to_string(QueueType value) {
  switch (value) {
    case QUEUE_TYPE_GRAPHICS:
      return "QUEUE_TYPE_GRAPHICS";
    case QUEUE_TYPE_TRANSFER:
      return "QUEUE_TYPE_TRANSFER";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown QueueType value.");
#endif
      return {};
  }
}

std::string QueueTypeName(QueueType value) {
  switch (value) {
    case QUEUE_TYPE_GRAPHICS:
      return "Graphics";
    case QUEUE_TYPE_TRANSFER:
      return "Transfer";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown QueueType value.");
#endif
      return {};
  }
}

std::string to_string(SubmissionStatus value) {
  switch (value) {
    case SUBMISSION_STATUS_PENDING:
      return "SUBMISSION_STATUS_PENDING";
    case SUBMISSION_STATUS_COMPLETED:
      return "SUBMISSION_STATUS_COMPLETED";
    case SUBMISSION_STATUS_FAILED:
      return "SUBMISSION_STATUS_FAILED";
    case SUBMISSION_STATUS_CANCELED:
      return "SUBMISSION_STATUS_CANCELED";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown SubmissionStatus value.");
#endif
      return {};
  }
}

std::string SubmissionStatusName(SubmissionStatus value) {
  switch (value) {
    case SUBMISSION_STATUS_PENDING:
      return "Pending";
    case SUBMISSION_STATUS_COMPLETED:
      return "Completed";
    case SUBMISSION_STATUS_FAILED:
      return "Failed";
    case SUBMISSION_STATUS_CANCELED:
      return "Canceled";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown SubmissionStatus value.");
#endif
      return {};
  }
}

std::string to_string(ImageFormat value) {
  switch (value) {
    case IMAGE_FORMAT_UNKNOWN:
      return "IMAGE_FORMAT_UNKNOWN";
    case IMAGE_FORMAT_R8_UINT:
      return "IMAGE_FORMAT_R8_UINT";
    case IMAGE_FORMAT_R8_INT:
      return "IMAGE_FORMAT_R8_INT";
    case IMAGE_FORMAT_R8_UNORM:
      return "IMAGE_FORMAT_R8_UNORM";
    case IMAGE_FORMAT_R8G8_UINT:
      return "IMAGE_FORMAT_R8G8_UINT";
    case IMAGE_FORMAT_R8G8_INT:
      return "IMAGE_FORMAT_R8G8_INT";
    case IMAGE_FORMAT_R8G8_UNORM:
      return "IMAGE_FORMAT_R8G8_UNORM";
    case IMAGE_FORMAT_R8G8B8_UINT:
      return "IMAGE_FORMAT_R8G8B8_UINT";
    case IMAGE_FORMAT_R8G8B8_INT:
      return "IMAGE_FORMAT_R8G8B8_INT";
    case IMAGE_FORMAT_R8G8B8_UNORM:
      return "IMAGE_FORMAT_R8G8B8_UNORM";
    case IMAGE_FORMAT_R8G8B8A8_UINT:
      return "IMAGE_FORMAT_R8G8B8A8_UINT";
    case IMAGE_FORMAT_R8G8B8A8_INT:
      return "IMAGE_FORMAT_R8G8B8A8_INT";
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return "IMAGE_FORMAT_R8G8B8A8_UNORM";
    case IMAGE_FORMAT_R32_UINT:
      return "IMAGE_FORMAT_R32_UINT";
    case IMAGE_FORMAT_R32_INT:
      return "IMAGE_FORMAT_R32_INT";
    case IMAGE_FORMAT_R32_SFLOAT:
      return "IMAGE_FORMAT_R32_SFLOAT";
    case IMAGE_FORMAT_R32G32_UINT:
      return "IMAGE_FORMAT_R32G32_UINT";
    case IMAGE_FORMAT_R32G32_INT:
      return "IMAGE_FORMAT_R32G32_INT";
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return "IMAGE_FORMAT_R32G32_SFLOAT";
    case IMAGE_FORMAT_R32G32B32_UINT:
      return "IMAGE_FORMAT_R32G32B32_UINT";
    case IMAGE_FORMAT_R32G32B32_INT:
      return "IMAGE_FORMAT_R32G32B32_INT";
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return "IMAGE_FORMAT_R32G32B32_SFLOAT";
    case IMAGE_FORMAT_R32G32B32A32_UINT:
      return "IMAGE_FORMAT_R32G32B32A32_UINT";
    case IMAGE_FORMAT_R32G32B32A32_INT:
      return "IMAGE_FORMAT_R32G32B32A32_INT";
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return "IMAGE_FORMAT_R32G32B32A32_SFLOAT";
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return "IMAGE_FORMAT_B8G8R8A8_UNORM";
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
      return "IMAGE_FORMAT_R16G16B16A16_SFLOAT";
    case IMAGE_FORMAT_R10G10B10A2_UNORM:
      return "IMAGE_FORMAT_R10G10B10A2_UNORM";
    case IMAGE_FORMAT_B10G10R10A2_UNORM:
      return "IMAGE_FORMAT_B10G10R10A2_UNORM";
    case IMAGE_FORMAT_D16_UNORM:
      return "IMAGE_FORMAT_D16_UNORM";
    case IMAGE_FORMAT_D32_SFLOAT:
      return "IMAGE_FORMAT_D32_SFLOAT";
    case IMAGE_FORMAT_D24_UNORM_S8_UINT:
      return "IMAGE_FORMAT_D24_UNORM_S8_UINT";
    default:
      return "UNKNOWN";
  }
}

std::string ImageFormatName(ImageFormat value) {
  switch (value) {
    case IMAGE_FORMAT_UNKNOWN:
      return "UNKNOWN";
    case IMAGE_FORMAT_R8_UINT:
      return "R8_UINT";
    case IMAGE_FORMAT_R8_INT:
      return "R8_INT";
    case IMAGE_FORMAT_R8_UNORM:
      return "R8_UNORM";
    case IMAGE_FORMAT_R8G8_UINT:
      return "R8G8_UINT";
    case IMAGE_FORMAT_R8G8_INT:
      return "R8G8_INT";
    case IMAGE_FORMAT_R8G8_UNORM:
      return "R8G8_UNORM";
    case IMAGE_FORMAT_R8G8B8_UINT:
      return "R8G8B8_UINT";
    case IMAGE_FORMAT_R8G8B8_INT:
      return "R8G8B8_INT";
    case IMAGE_FORMAT_R8G8B8_UNORM:
      return "R8G8B8_UNORM";
    case IMAGE_FORMAT_R8G8B8A8_UINT:
      return "R8G8B8A8_UINT";
    case IMAGE_FORMAT_R8G8B8A8_INT:
      return "R8G8B8A8_INT";
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return "R8G8B8A8_UNORM";
    case IMAGE_FORMAT_R32_UINT:
      return "R32_UINT";
    case IMAGE_FORMAT_R32_INT:
      return "R32_INT";
    case IMAGE_FORMAT_R32_SFLOAT:
      return "R32_SFLOAT";
    case IMAGE_FORMAT_R32G32_UINT:
      return "R32G32_UINT";
    case IMAGE_FORMAT_R32G32_INT:
      return "R32G32_INT";
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return "R32G32_SFLOAT";
    case IMAGE_FORMAT_R32G32B32_UINT:
      return "R32G32B32_UINT";
    case IMAGE_FORMAT_R32G32B32_INT:
      return "R32G32B32_INT";
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return "R32G32B32_SFLOAT";
    case IMAGE_FORMAT_R32G32B32A32_UINT:
      return "R32G32B32A32_UINT";
    case IMAGE_FORMAT_R32G32B32A32_INT:
      return "R32G32B32A32_INT";
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return "R32G32B32A32_SFLOAT";
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return "B8G8R8A8_UNORM";
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
      return "R16G16B16A16_SFLOAT";
    case IMAGE_FORMAT_R10G10B10A2_UNORM:
      return "R10G10B10A2_UNORM";
    case IMAGE_FORMAT_B10G10R10A2_UNORM:
      return "B10G10R10A2_UNORM";
    case IMAGE_FORMAT_D16_UNORM:
      return "D16_UNORM";
    case IMAGE_FORMAT_D32_SFLOAT:
      return "D32_SFLOAT";
    case IMAGE_FORMAT_D24_UNORM_S8_UINT:
      return "D24_UNORM_S8_UINT";
    default:
      return "UNKNOWN";
  }
}

bool IsSignedIntegerFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_R8_INT:
    case IMAGE_FORMAT_R8G8_INT:
    case IMAGE_FORMAT_R8G8B8_INT:
    case IMAGE_FORMAT_R8G8B8A8_INT:
    case IMAGE_FORMAT_R32_INT:
    case IMAGE_FORMAT_R32G32_INT:
    case IMAGE_FORMAT_R32G32B32_INT:
    case IMAGE_FORMAT_R32G32B32A32_INT:
      return true;
    default:
      return false;
  }
}

bool IsUnsignedIntegerFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_R8_UINT:
    case IMAGE_FORMAT_R8G8_UINT:
    case IMAGE_FORMAT_R8G8B8_UINT:
    case IMAGE_FORMAT_R8G8B8A8_UINT:
    case IMAGE_FORMAT_R32_UINT:
    case IMAGE_FORMAT_R32G32_UINT:
    case IMAGE_FORMAT_R32G32B32_UINT:
    case IMAGE_FORMAT_R32G32B32A32_UINT:
      return true;
    default:
      return false;
  }
}

std::string to_string(ShaderStage stage) {
  switch (stage) {
    case SHADER_STAGE_VERTEX:
      return "SHADER_STAGE_VERTEX";
    case SHADER_STAGE_FRAGMENT:
      return "SHADER_STAGE_FRAGMENT";
    case SHADER_STAGE_COMPUTE:
      return "SHADER_STAGE_COMPUTE";
    case SHADER_STAGE_RAY_GENERATION:
      return "SHADER_STAGE_RAY_GENERATION";
    case SHADER_STAGE_MISS:
      return "SHADER_STAGE_MISS";
    case SHADER_STAGE_CLOSEST_HIT:
      return "SHADER_STAGE_CLOSEST_HIT";
    case SHADER_STAGE_ANY_HIT:
      return "SHADER_STAGE_ANY_HIT";
    case SHADER_STAGE_INTERSECTION:
      return "SHADER_STAGE_INTERSECTION";
    case SHADER_STAGE_CALLABLE:
      return "SHADER_STAGE_CALLABLE";
  }
  return "UNKNOWN";
}

uint32_t InputTypeSize(InputType type) {
  switch (type) {
    case INPUT_TYPE_FLOAT:
      return 4;
    case INPUT_TYPE_FLOAT2:
      return 8;
    case INPUT_TYPE_FLOAT3:
      return 12;
    case INPUT_TYPE_FLOAT4:
      return 16;
    case INPUT_TYPE_INT:
      return 4;
    case INPUT_TYPE_INT2:
      return 8;
    case INPUT_TYPE_INT3:
      return 12;
    case INPUT_TYPE_INT4:
      return 16;
    case INPUT_TYPE_UINT:
      return 4;
    case INPUT_TYPE_UINT2:
      return 8;
    case INPUT_TYPE_UINT3:
      return 12;
    case INPUT_TYPE_UINT4:
      return 16;
  }
  return 0;
}

std::string to_string(InputType type) {
  switch (type) {
    case INPUT_TYPE_FLOAT:
      return "INPUT_TYPE_FLOAT";
    case INPUT_TYPE_FLOAT2:
      return "INPUT_TYPE_FLOAT2";
    case INPUT_TYPE_FLOAT3:
      return "INPUT_TYPE_FLOAT3";
    case INPUT_TYPE_FLOAT4:
      return "INPUT_TYPE_FLOAT4";
    case INPUT_TYPE_INT:
      return "INPUT_TYPE_INT";
    case INPUT_TYPE_INT2:
      return "INPUT_TYPE_INT2";
    case INPUT_TYPE_INT3:
      return "INPUT_TYPE_INT3";
    case INPUT_TYPE_INT4:
      return "INPUT_TYPE_INT4";
    case INPUT_TYPE_UINT:
      return "INPUT_TYPE_UINT";
    case INPUT_TYPE_UINT2:
      return "INPUT_TYPE_UINT2";
    case INPUT_TYPE_UINT3:
      return "INPUT_TYPE_UINT3";
    case INPUT_TYPE_UINT4:
      return "INPUT_TYPE_UINT4";
  }
  return "UNKNOWN";
}

std::string to_string(AttachmentLoadOp op) {
  switch (op) {
    case ATTACHMENT_LOAD_OP_LOAD:
      return "Load";
    case ATTACHMENT_LOAD_OP_CLEAR:
      return "Clear";
    case ATTACHMENT_LOAD_OP_DISCARD:
      return "Discard";
  }
  return "Unknown";
}

std::string to_string(AttachmentStoreOp op) {
  switch (op) {
    case ATTACHMENT_STORE_OP_STORE:
      return "Store";
    case ATTACHMENT_STORE_OP_DISCARD:
      return "Discard";
  }
  return "Unknown";
}

std::string to_string(IndexFormat format) {
  switch (format) {
    case INDEX_FORMAT_UINT16:
      return "INDEX_FORMAT_UINT16";
    case INDEX_FORMAT_UINT32:
      return "INDEX_FORMAT_UINT32";
  }
  return "UNKNOWN";
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

BlendState::BlendState(bool enabled) : blend_enable(enabled) {
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

std::string to_string(BlendFactor value) {
  const char *names[] = {"Zero",
                         "One",
                         "Source Color",
                         "One Minus Source Color",
                         "Destination Color",
                         "One Minus Destination Color",
                         "Source Alpha",
                         "One Minus Source Alpha",
                         "Destination Alpha",
                         "One Minus Destination Alpha"};
  if (value < BLEND_FACTOR_ZERO || value > BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
    return "Unknown";
  return names[value];
}

std::string to_string(BlendOp value) {
  const char *names[] = {"Add", "Subtract", "Reverse Subtract", "Min", "Max"};
  if (value < BLEND_OP_ADD || value > BLEND_OP_MAX)
    return "Unknown";
  return names[value];
}

}  // namespace grassland::rhi

namespace grassland::rhi {
std::string to_string(AccelerationStructureType value) {
  switch (value) {
    case ACCELERATION_STRUCTURE_BOTTOM_LEVEL:
      return "ACCELERATION_STRUCTURE_BOTTOM_LEVEL";
    case ACCELERATION_STRUCTURE_TOP_LEVEL:
      return "ACCELERATION_STRUCTURE_TOP_LEVEL";
    default:
      return "UNKNOWN";
  }
}

std::string to_string(AccelerationStructureBuildMode value) {
  switch (value) {
    case ACCELERATION_STRUCTURE_BUILD:
      return "ACCELERATION_STRUCTURE_BUILD";
    case ACCELERATION_STRUCTURE_UPDATE:
      return "ACCELERATION_STRUCTURE_UPDATE";
    default:
      return "UNKNOWN";
  }
}
}  // namespace grassland::rhi
