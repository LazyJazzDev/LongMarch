#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

class Entity : public Object {
 public:
  Entity(Core *core);

  Core *GetCore() const;

  ~Entity() override = default;

  operator bool() const {
    return core_ != nullptr;
  }

 protected:
  Core *core_;
};

}  // namespace sparkium
