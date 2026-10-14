#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <type_traits>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
namespace {

class TextObject final : public Object<IObject> {
 public:
  explicit TextObject(std::atomic<int> &destructions) : destructions_(destructions) {
  }

  ~TextObject() override {
    ++destructions_;
  }

  const std::string &Text() const {
    return text_;
  }

 private:
  std::atomic<int> &destructions_;
  std::string text_{"typed backend object"};
};

static_assert(sizeof(ObjectPtr<TextObject>) == sizeof(TextObject *));
static_assert(std::is_constructible_v<ObjectPtr<IObject>, const ObjectPtr<TextObject> &>);
static_assert(std::is_constructible_v<ObjectPtr<IObject>, ObjectPtr<TextObject> &&>);
static_assert(!std::is_constructible_v<ObjectPtr<TextObject>, ObjectPtr<IObject>>);
static_assert(!std::is_constructible_v<ObjectPtr<TextObject>, TextObject *>);

TEST(RhiObjectPtr, TypedAccessCopyMoveAndSelfAssignmentKeepTheSameObjectAlive) {
  std::atomic<int> destructions{0};
  auto owner = ObjectPtr<TextObject>::Adopt(new TextObject(destructions));
  EXPECT_EQ(owner->Text(), "typed backend object");
  EXPECT_EQ((*owner).Text(), owner->Text());
  auto borrowed = owner.Get();
  auto retained = ObjectPtr<TextObject>::Retain(borrowed);
  auto copy = owner;
  auto moved = std::move(copy);
  EXPECT_FALSE(copy);
  EXPECT_EQ(moved.Get(), borrowed);
  auto &self = owner;
  owner = self;
  owner = std::move(self);
  EXPECT_EQ(owner.Get(), borrowed);
  owner.Reset();
  moved = {};
  EXPECT_EQ(destructions, 0);
  EXPECT_EQ(retained->Text(), "typed backend object");
  retained.Reset();
  EXPECT_EQ(destructions, 1);
}

TEST(RhiObjectPtr, UpcastsReplacementAndDetachTransferExactlyOneReference) {
  std::atomic<int> destructions{0};
  auto derived = ObjectPtr<TextObject>::Adopt(new TextObject(destructions));
  ObjectPtr<IObject> base = derived;
  ObjectPtr<IObject> moved = std::move(derived);
  EXPECT_FALSE(derived);
  base.Reset();
  auto replacement = ObjectPtr<IObject>::Adopt(new TextObject(destructions));
  replacement = std::move(moved);
  EXPECT_FALSE(moved);
  EXPECT_EQ(destructions, 1);
  auto raw = replacement.Detach();
  EXPECT_FALSE(replacement);
  EXPECT_EQ(destructions, 1);
  auto adopted = ObjectPtr<IObject>::Adopt(raw);
  adopted.Reset();
  EXPECT_EQ(destructions, 2);
  EXPECT_FALSE(ObjectPtr<IObject>::Retain(nullptr));
  EXPECT_FALSE(ObjectPtr<IObject>::Adopt(nullptr));
}

TEST(RhiObjectPtr, IndependentThreadHandlesShareAtomicOwnership) {
  std::atomic<int> destructions{0};
  auto owner = ObjectPtr<TextObject>::Adopt(new TextObject(destructions));
  std::thread threads[4];
  for (auto &thread : threads) {
    thread = std::thread([ref = owner] {
      for (int i = 0; i < 1000; ++i) {
        ObjectPtr<IObject> local = ref;
      }
    });
  }
  owner.Reset();
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(destructions, 1);
}

}  // namespace
}  // namespace grassland::rhi::backend
