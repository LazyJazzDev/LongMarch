#include "grassland/rhi/rhi_util.h"

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi {

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
    case IMAGE_FORMAT_RGBA8_UNORM:
      return "IMAGE_FORMAT_RGBA8_UNORM";
    case IMAGE_FORMAT_BGRA8_UNORM:
      return "IMAGE_FORMAT_BGRA8_UNORM";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown ImageFormat value.");
#endif
      return {};
  }
}

std::string ImageFormatName(ImageFormat value) {
  switch (value) {
    case IMAGE_FORMAT_UNKNOWN:
      return "Unknown";
    case IMAGE_FORMAT_RGBA8_UNORM:
      return "RGBA8 UNORM";
    case IMAGE_FORMAT_BGRA8_UNORM:
      return "BGRA8 UNORM";
    default:
#if !defined(NDEBUG)
      ReportError(-1, "Unknown ImageFormat value.");
#endif
      return {};
  }
}

}  // namespace grassland::rhi
