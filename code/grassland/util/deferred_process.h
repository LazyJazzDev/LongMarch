#pragma once
#include "functional"

namespace CD {
class DeferredProcess {
 public:
  DeferredProcess(const std::function<void()> &func);
  DeferredProcess(const DeferredProcess &) = delete;
  DeferredProcess &operator=(const DeferredProcess &) = delete;
  DeferredProcess(DeferredProcess &&) = delete;
  DeferredProcess &operator=(DeferredProcess &&) = delete;
  ~DeferredProcess();

 private:
  std::function<void()> func_;
};
}  // namespace CD
