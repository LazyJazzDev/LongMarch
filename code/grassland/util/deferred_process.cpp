#include "grassland/util/deferred_process.h"

namespace CD {

DeferredProcess::DeferredProcess(const std::function<void()> &func) : func_(func) {
}

DeferredProcess::~DeferredProcess() {
  if (func_) {
    func_();
  }
}

}  // namespace CD
