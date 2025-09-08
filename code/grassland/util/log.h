#pragma once
#if !defined(__CUDACC__)
#include "fmt/format.h"
#else
namespace fmt {
template <typename... Args>
std::string format(const std::string &format_str, Args &&...args) {
  return "";
}
}  // namespace fmt
#endif
#include "iostream"

namespace CD {
std::string GetTimestamp();

void LogInfo(const std::string &message);

void LogWarning(const std::string &message);

void LogError(const std::string &message);

template <class... Args>
void LogInfo(const std::string &message, Args &&...args) {
  LogInfo(fmt::format(message, std::forward<Args>(args)...));
}

template <class... Args>
void LogWarning(const std::string &message, Args &&...args) {
  LogWarning(fmt::format(message, std::forward<Args>(args)...));
}

template <class... Args>
void LogError(const std::string &message, Args &&...args) {
  LogError(fmt::format(message, std::forward<Args>(args)...));
}
}  // namespace CD
