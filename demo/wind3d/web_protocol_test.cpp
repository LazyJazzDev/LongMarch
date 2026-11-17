#include "web_protocol.h"

#include <atomic>
#include <iostream>

int main() {
  namespace fs = std::filesystem;
  using namespace wind3d::web;
  auto directory = fs::temp_directory_path() /
                   ("wind3d-publish-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directory(directory);
  auto path = directory / "frame.bin";
  bool passed = Write(path, "old complete packet");
  std::ifstream reader(path, std::ios::binary);
  std::atomic<bool> entered{false};
  bool written = false;
  std::thread writer([&] {
    entered = true;
    written = Write(path, "new complete packet");
  });
  while (!entered)
    std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
  std::string old((std::istreambuf_iterator<char>(reader)), {});
  reader.close();
  writer.join();
  std::ifstream latest(path, std::ios::binary);
  std::string current((std::istreambuf_iterator<char>(latest)), {});
  latest.close();
  passed &= written && old == "old complete packet" && current == "new complete packet";
  fs::remove(path);
  fs::remove(directory);
  std::cout << (passed ? "PASS" : "FAIL") << ": atomic packet publication with an overlapping reader\n";
  return passed ? 0 : 1;
}
