#pragma once
#include <stdexcept>
#include <string>

namespace sparkium::backend {

[[noreturn]] inline void NativeUnsupported() {
  throw std::runtime_error(
      "operation unavailable on this headless native backend; use a graphics backend for rasterization or presentation");
}

}  // namespace sparkium::backend
