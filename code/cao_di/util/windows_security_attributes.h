#pragma once
#ifdef _WIN64
#include "cao_di/util/util_util.h"

namespace CD {

class WindowsSecurityAttributes {
 protected:
  SECURITY_ATTRIBUTES win_security_attributes_;
  PSECURITY_DESCRIPTOR win_p_security_descriptor_;

 public:
  WindowsSecurityAttributes();
  SECURITY_ATTRIBUTES *operator&();
  ~WindowsSecurityAttributes();
};

}  // namespace CD

#endif
