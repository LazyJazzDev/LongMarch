#include "grassland/util/vendor_id.h"

#include "fmt/format.h"

namespace CD {

std::string PCIVendorIDToName(uint32_t vendor_id) {
  switch (vendor_id) {
    case 0x1002:
      return "AMD (ATI)";
    case 0x1022:
      return "AMD";
    case 0x10DE:
      return "NVIDIA";
    case 0x8086:
      return "Intel";
    case 0x106B:
      return "Apple";
    case 0x5143:
    case 0x17CB:
      return "Qualcomm";
    case 0x19E5:
      return "HUAWEI";
    case 0x13B5:
      return "ARM";
    default:
      return fmt::format("Unknown vendor id: 0x{:04X}", vendor_id);
      // Rewrite with std::string
  }
}

}  // namespace CD
