#pragma once
#include <cstdint>
#include <optional>
#include <string_view>

namespace grassland::rhi {
struct InterfaceID {
  uint64_t high{};
  uint64_t low{};

  constexpr bool operator==(InterfaceID other) const noexcept {
    return high == other.high && low == other.low;
  }

  constexpr bool operator!=(InterfaceID other) const noexcept {
    return !(*this == other);
  }
};

// Canonical UUID spelling, interpreted in textual order, independent of host endianness.
// This is a value representation, not the Windows GUID memory layout.
constexpr std::optional<InterfaceID> ParseInterfaceID(std::string_view text) noexcept {
  if (text.size() != 36)
    return std::nullopt;
  InterfaceID result;
  unsigned digits = 0;
  for (unsigned i = 0; i < text.size(); ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (text[i] != '-')
        return std::nullopt;
      continue;
    }
    const char c = text[i];
    unsigned value{};
    if (c >= '0' && c <= '9')
      value = unsigned(c - '0');
    else if (c >= 'a' && c <= 'f')
      value = unsigned(c - 'a') + 10;
    else if (c >= 'A' && c <= 'F')
      value = unsigned(c - 'A') + 10;
    else
      return std::nullopt;
    auto &half = digits++ < 16 ? result.high : result.low;
    half = (half << 4) | value;
  }
  return result;
}

template <typename T>
constexpr InterfaceID uuid() = delete;
}  // namespace grassland::rhi

// Use at global namespace scope, before the first query for Type. No runtime registration.
#define LONGMARCH_RHI_UUID(Type, Text)                                                               \
  template <>                                                                                        \
  inline constexpr grassland::rhi::InterfaceID grassland::rhi::uuid<Type>() {                        \
    static_assert(grassland::rhi::ParseInterfaceID(Text).has_value(), "Invalid RHI interface UUID"); \
    return *grassland::rhi::ParseInterfaceID(Text);                                                  \
  }
