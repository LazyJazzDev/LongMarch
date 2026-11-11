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

#define LONGMARCH_RHI_QUERY_MULTIPLE(Type, Base, Side)                   \
  using Base::QueryInterface;                                            \
  bool QueryInterface(InterfaceID id, void **output) noexcept override { \
    if (!output)                                                         \
      return false;                                                      \
    if (auto result = MatchInterfaces<Type, Side>(id, this)) {           \
      *output = result;                                                  \
      AddRef();                                                          \
      return true;                                                       \
    }                                                                    \
    return Base::QueryInterface(id, output);                             \
  }                                                                      \
  void AddRef() noexcept override {                                      \
    Base::AddRef();                                                      \
  }                                                                      \
  void Release() noexcept override {                                     \
    Base::Release();                                                     \
  }
