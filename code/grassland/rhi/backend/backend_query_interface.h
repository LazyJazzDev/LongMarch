#pragma once

// The leaf chooses the canonical IObject path and all supported interfaces.
#define LONGMARCH_RHI_FINAL_QUERY(Primary, ...)                                            \
  template <typename T>                                                                    \
  T *QueryInterface() noexcept {                                                           \
    void *result = nullptr;                                                                \
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;        \
  }                                                                                        \
  bool QueryInterface(InterfaceID id, void **output) noexcept override {                   \
    if (!output)                                                                           \
      return false;                                                                        \
    *output = id == uuid<IObject>() ? static_cast<IObject *>(static_cast<Primary *>(this)) \
                                    : MatchInterfaces<__VA_ARGS__>(id, this);              \
    if (!*output)                                                                          \
      return false;                                                                        \
    AddRef();                                                                              \
    return true;                                                                           \
  }
