#include <atomic>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

class IObject {
 public:
  void AddRef() noexcept {
    counter_.fetch_add(1, std::memory_order_relaxed);
  }

  void Release() noexcept {
    if (counter_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

 protected:
  IObject() = default;
  virtual ~IObject() = default;

 private:
  // A new object's initial reference is transferred by Adopt.
  std::atomic<uint32_t> counter_{1};
};

class ObjectRef {
 public:
  ObjectRef() noexcept = default;

  // Take an existing reference without incrementing the count.
  static ObjectRef Adopt(IObject *object) noexcept {
    ObjectRef ref;
    ref.ptr_ = object;
    return ref;
  }

  // The caller must already guarantee that the borrowed object is alive.
  static ObjectRef Retain(IObject *object) noexcept {
    if (object) {
      object->AddRef();
    }
    return Adopt(object);
  }

  ObjectRef(const ObjectRef &other) noexcept : ObjectRef(Retain(other.ptr_)) {
  }

  ObjectRef(ObjectRef &&other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {
  }

  ObjectRef &operator=(const ObjectRef &other) noexcept {
    ObjectRef copy(other);
    std::swap(ptr_, copy.ptr_);
    return *this;
  }

  ObjectRef &operator=(ObjectRef &&other) noexcept {
    if (this != &other) {
      ObjectRef moved(std::move(other));
      std::swap(ptr_, moved.ptr_);
    }
    return *this;
  }

  ~ObjectRef() {
    if (ptr_) {
      ptr_->Release();
    }
  }

  explicit operator bool() const noexcept {
    return ptr_ != nullptr;
  }

 private:
  IObject *ptr_ = nullptr;
};

class ExampleObject final : public IObject {
 public:
  ExampleObject(std::string text, std::atomic<uint32_t> &destructions)
      : text_(std::move(text)),
        destructions_(destructions) {
  }

  const std::string &GetText() const noexcept {
    return text_;
  }

 private:
  ~ExampleObject() override {
    ++destructions_;
  }

  std::string text_;
  std::atomic<uint32_t> &destructions_;
};

int main() {
  std::atomic<uint32_t> destructions{0};
  auto *object = new ExampleObject("Hello from ExampleObject!", destructions);
  auto owner = ObjectRef::Adopt(object);
  auto retained = ObjectRef::Retain(object);
  ObjectRef copy = owner;
  ObjectRef moved = std::move(copy);
  if (copy || !moved || ObjectRef{} || ObjectRef::Retain(nullptr)) {
    return 1;
  }

  retained = owner;  // Assignment between references to the same object.
  auto &self = owner;
  owner = self;
  owner = std::move(self);  // Self-move preserves the reference.
  retained = std::move(moved);
  if (!owner || !retained || moved) {
    return 1;
  }

  auto replaced = ObjectRef::Adopt(new ExampleObject("Temporary object", destructions));
  replaced = owner;  // Release the previous, different object.
  if (destructions != 1) {
    return 1;
  }
  replaced = {};

  // Each thread owns a separate handle; only the object's count is shared.
  std::thread threads[4];
  for (auto &thread : threads) {
    thread = std::thread([ref = owner] {
      for (int i = 0; i < 10000; ++i) {
        ObjectRef local = ref;
      }
    });
  }
  owner = {};
  for (auto &thread : threads) {
    thread.join();
  }
  if (destructions != 1) {
    return 1;
  }
  // retained still owns the object after owner and the thread handles are released.
  std::cout << "Stored text: " << object->GetText() << '\n';
  retained = {};
  if (destructions != 2) {
    return 1;
  }

  std::cout << "ObjectRef: adopt/retain, copy/move and concurrent ownership passed.\n"
            << "Derived objects destroyed exactly once: " << destructions << '\n';
}
