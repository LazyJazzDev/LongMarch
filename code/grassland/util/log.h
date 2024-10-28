#pragma once
#include "iostream"
#if !defined(__CUDACC__)
#include "spdlog/spdlog.h"
#endif

namespace grassland {

std::string GetTimestamp();

void LogInfo(const std::string &message);

void LogWarning(const std::string &message);

void LogError(const std::string &message);

template <class... Args>
void LogInfo(const std::string &message, Args &&...args) {
#if !defined(__CUDACC__)
  spdlog::info(message, std::forward<Args>(args)...);
#endif
}

template <class... Args>
void LogWarning(const std::string &message, Args &&...args) {
#if !defined(__CUDACC__)
  spdlog::warn(message, std::forward<Args>(args)...);
#endif
}

template <class... Args>
void LogError(const std::string &message, Args &&...args) {
#if !defined(__CUDACC__)
  spdlog::error(message, std::forward<Args>(args)...);
#endif
}
}  // namespace grassland
