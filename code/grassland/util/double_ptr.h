#pragma once

#include <memory>

namespace grassland {

enum class double_ptr_type { raw = 0, shared, unique };

template <class ContentType>
class double_ptr {
 public:
  double_ptr(nullptr_t) {
    raw_ptr = nullptr;
    type = double_ptr_type::raw;
  }

  double_ptr(ContentType **ptr) {
    raw_ptr = ptr;
    type = double_ptr_type::raw;
  }

  double_ptr(std::shared_ptr<ContentType> *ptr) {
    shared_ptr = ptr;
    type = double_ptr_type::shared;
  }

  double_ptr(std::unique_ptr<ContentType> *ptr) {
    unique_ptr = ptr;
    type = double_ptr_type::unique;
  }

  ContentType *operator=(ContentType *ptr) {
    switch (type) {
      case double_ptr_type::raw:
        *raw_ptr = ptr;
        break;
      case double_ptr_type::shared:
        *shared_ptr = std::shared_ptr<ContentType>(ptr);
        break;
      case double_ptr_type::unique:
        *unique_ptr = std::unique_ptr<ContentType>(ptr);
        break;
    }
    return ptr;
  }

  template <class... Args>
  ContentType *construct(Args &&...args) {
    switch (type) {
      case double_ptr_type::raw:
        *raw_ptr = new ContentType(std::forward<Args>(args)...);
        break;
      case double_ptr_type::shared:
        *shared_ptr =
            std::make_shared<ContentType>(std::forward<Args>(args)...);
        break;
      case double_ptr_type::unique:
        *unique_ptr =
            std::make_unique<ContentType>(std::forward<Args>(args)...);
        break;
    }
    return operator ContentType *();
  }

  explicit operator ContentType *() {
    switch (type) {
      case double_ptr_type::raw:
        return *raw_ptr;
      case double_ptr_type::shared:
        return shared_ptr->get();
      case double_ptr_type::unique:
        return unique_ptr->get();
    }
    return nullptr;
  }

  ContentType *operator->() {
    return operator ContentType *();
  }

  ContentType *operator*() {
    return operator ContentType *();
  }

  operator bool() {
    return raw_ptr != nullptr;
  }

 private:
  union {
    std::shared_ptr<ContentType> *shared_ptr;
    std::unique_ptr<ContentType> *unique_ptr;
    ContentType **raw_ptr;
  };
  double_ptr_type type{double_ptr_type::raw};
};
}  // namespace grassland
