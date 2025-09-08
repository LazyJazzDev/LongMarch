#include "grassland/util/log.h"

#include "chrono"
#include "iomanip"
#include "sstream"

namespace CD {
std::string GetTimestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t current_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
#ifdef _WIN64
  localtime_s(&local_time, &current_time);
#else
  localtime_r(&current_time, &local_time);
#endif
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  std::ostringstream oss;
  oss << std::put_time(&local_time, "[%Y-%m-%d %H:%M:%S.") << std::setfill('0') << std::setw(3) << milliseconds.count()
      << "]";
  return oss.str();
}

void LogInfo(const std::string &message) {
  std::cout << "[INFO][" << GetTimestamp() << "] " << message << std::endl;
}

void LogWarning(const std::string &message) {
  std::cout << "[WARNING][" << GetTimestamp() << "] " << message << std::endl;
}

void LogError(const std::string &message) {
  std::cout << "[ERROR][" << GetTimestamp() << "] " << message << std::endl;
}
}  // namespace CD
