#pragma once
#include <memory>

namespace grassland::utils {

enum class double_ptr_type { raw = 0, shared, unique };

template <class ContentType>
class double_ptr {
 public:
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

  operator ContentType *() {
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
}  // namespace grassland::utils
