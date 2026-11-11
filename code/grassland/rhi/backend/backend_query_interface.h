#pragma once

// The primary base selects the canonical IObject identity. Additional implementation
// interfaces share that base's reference count, even with non-virtual inheritance.
#define LONGMARCH_RHI_QUERY(Type, Base)                                  \
  using Base::QueryInterface;                                            \
  bool QueryInterface(InterfaceID id, void **output) noexcept override { \
    if (!output)                                                         \
      return false;                                                      \
    if (auto result = MatchInterface<Type>(id, this)) {                  \
      *output = result;                                                  \
      AddRef();                                                          \
      return true;                                                       \
    }                                                                    \
    return Base::QueryInterface(id, output);                             \
  }

#define LONGMARCH_RHI_QUERY_MULTIPLE(Type, Base, Side)                      \
  template <typename Interface>                                             \
  Interface *QueryInterface() noexcept {                                    \
    return static_cast<Base *>(this)->template QueryInterface<Interface>(); \
  }                                                                         \
  bool QueryInterface(InterfaceID id, void **output) noexcept override {    \
    if (!output)                                                            \
      return false;                                                         \
    if (auto result = MatchInterfaces<Type, Side>(id, this)) {              \
      *output = result;                                                     \
      AddRef();                                                             \
      return true;                                                          \
    }                                                                       \
    return Base::QueryInterface(id, output);                                \
  }                                                                         \
  void AddRef() noexcept override {                                         \
    Base::AddRef();                                                         \
  }                                                                         \
  void Release() noexcept override {                                        \
    Base::Release();                                                        \
  }

// The final implementation explicitly lists every supported ancestor interface.
#define LONGMARCH_RHI_QUERY_INTERFACES(Base, ...)                        \
  using Base::QueryInterface;                                            \
  bool QueryInterface(InterfaceID id, void **output) noexcept override { \
    if (!output)                                                         \
      return false;                                                      \
    if (auto result = MatchInterfaces<__VA_ARGS__>(id, this)) {          \
      *output = result;                                                  \
      AddRef();                                                          \
      return true;                                                       \
    }                                                                    \
    return Base::QueryInterface(id, output);                             \
  }

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
