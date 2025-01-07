#include "grassland/util/deferred_process.h"

namespace grassland {

DeferredProcess::DeferredProcess(const std::function<void()> &func) : func_(func) {
}

DeferredProcess::~DeferredProcess() {
  if (func_) {
    func_();
  }
}

}  // namespace grassland
