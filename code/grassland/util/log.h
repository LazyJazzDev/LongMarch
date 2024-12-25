#pragma once
#include "fmt/format.h"
#include "iostream"

namespace grassland {

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
}  // namespace grassland
