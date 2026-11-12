#include <gtest/gtest.h>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhi_buffer.h"

namespace query_test {
class ILeft : public grassland::rhi::IObject {};

class IRight : public grassland::rhi::IObject {};

class IExtended : public ILeft {};

class ILeaf : public IExtended {};

class ISibling : public ILeft {};

}  // namespace query_test

LONGMARCH_RHI_INTERFACE(query_test::ILeft, grassland::rhi::IObject, "523d719e-fb75-41e8-addc-fc6abcf4575f")
LONGMARCH_RHI_INTERFACE(query_test::IRight, grassland::rhi::IObject, "0ae16d45-31b1-4981-bba0-90a7df182c7e")

LONGMARCH_RHI_INTERFACE(query_test::IExtended, query_test::ILeft, "beff92a1-db2c-47cb-80d9-0abff454b583")

LONGMARCH_RHI_INTERFACE(query_test::ILeaf, query_test::IExtended, "8cd4baff-b971-4c51-8001-2ca20a777261")
LONGMARCH_RHI_INTERFACE(query_test::ISibling, query_test::ILeft, "34328aa3-b1fb-4968-9073-886157bbffb4")

namespace query_test {
using namespace grassland::rhi;
using namespace grassland::rhi::backend;

class Diamond final : public Object<IExtended, IRight> {
 public:
  explicit Diamond(bool &destroyed) : destroyed_(destroyed) {
  }

  ~Diamond() override {
    destroyed_ = true;
  }

 private:
  bool &destroyed_;
};

class Extended final : public Object<ILeaf> {
 public:
};

class SideBuffer final : public Object<ILeft, IBuffer> {
 public:
  BufferDesc GetDesc() const override {
    return {4096};
  }
};

TEST(RhiQueryInterface, TypedPointerPreservesNoncanonicalInterfaceThroughCopyMoveAndRelease) {
  auto object = ObjPtr<SideBuffer>(new SideBuffer);
  auto buffer = static_cast<IBuffer *>(object.Get());
  ObjPtr<IObject> identity = buffer->QueryInterface<IObject>();
  ASSERT_NE(identity.Get(), static_cast<IObject *>(buffer));
  buffer->AddRef();
  grassland::rhi::ObjPtr<IBuffer> wrapped(buffer);
  auto copy = wrapped;
  auto moved = std::move(copy);
  EXPECT_EQ(copy.Get(), nullptr);
  wrapped.Release();
  EXPECT_EQ(wrapped.Get(), nullptr);
  EXPECT_EQ(moved.Get(), buffer);
  EXPECT_EQ(moved->GetDesc().size, 4096);
  ObjPtr<ILeft> side = moved->QueryInterface<ILeft>();
  EXPECT_EQ(side.Get(), static_cast<ILeft *>(object.Get()));
  moved.Release();
  side.Release();
  identity.Release();
  EXPECT_TRUE(object->IsUniquelyOwned());
}

static_assert(std::is_abstract_v<IExtended>);
static_assert(std::is_final_v<Diamond>);
static_assert(sizeof(IExtended) == sizeof(ILeft));

constexpr auto parsed = ParseInterfaceID("523d719e-fb75-41e8-addc-fc6abcf4575f");
static_assert(parsed.has_value());
static_assert(parsed->high == 0x523d719efb7541e8ULL);
static_assert(parsed->low == 0xaddcfc6abcf4575fULL);

static_assert(!ParseInterfaceID("523d719e-fb75-41e8-addc-fc6abcf4575x"));

class Branches final : public Object<ILeaf, ISibling> {};

TEST(RhiQueryInterface, SharedAncestorsUseTheFirstDeclaredInterfaceChain) {
  ObjPtr<Branches> object = new Branches;
  ObjPtr<ISibling> sibling = object->QueryInterface<ISibling>();
  ObjPtr<ILeft> ancestor = sibling->QueryInterface<ILeft>();
  EXPECT_EQ(ancestor.Get(), static_cast<ILeft *>(static_cast<ILeaf *>(object.Get())));
  EXPECT_NE(ancestor.Get(), static_cast<ILeft *>(sibling.Get()));
  ObjPtr<IExtended> middle = sibling->QueryInterface<IExtended>();
  EXPECT_EQ(middle.Get(), static_cast<IExtended *>(static_cast<ILeaf *>(object.Get())));
  ObjPtr<IObject> identity = sibling->QueryInterface<IObject>();
  EXPECT_EQ(identity.Get(), static_cast<IObject *>(static_cast<ILeaf *>(object.Get())));
  sibling.Release();
  ancestor.Release();
  middle.Release();
  identity.Release();
  EXPECT_TRUE(object->IsUniquelyOwned());
}

TEST(RhiQueryInterface, UUIDParsingIsStrictAndCaseIndependent) {
  EXPECT_EQ(ParseInterfaceID("523D719E-FB75-41E8-ADDC-FC6ABCF4575F"), parsed);
  for (const char *invalid : {"", "523d719efb7541e8addcfc6abcf4575f", "523d719e_fb75-41e8-addc-fc6abcf4575f",
                              "{523d719e-fb75-41e8-addc-fc6abcf4575f}", "523d719e-fb75-41e8-addc-fc6abcf4575f "})
    EXPECT_FALSE(ParseInterfaceID(invalid));
  EXPECT_EQ(*parsed, uuid<ILeft>());
  EXPECT_NE(uuid<ILeft>(), uuid<IRight>());
}

TEST(RhiQueryInterface, IndependentInterfacesHaveOneCounterAndCanonicalIdentity) {
  bool destroyed = false;
  ObjPtr<Diamond> object = new Diamond(destroyed);
  EXPECT_TRUE(object->IsUniquelyOwned());
  ObjPtr<ILeft> left = object->QueryInterface<ILeft>();
  ObjPtr<IRight> right = left->QueryInterface<IRight>();
  ASSERT_TRUE(left && right);
  EXPECT_FALSE(object->IsUniquelyOwned());
  EXPECT_NE(static_cast<void *>(left.Get()), static_cast<void *>(right.Get()));
  ObjPtr<IObject> left_identity = left->QueryInterface<IObject>();
  ObjPtr<IObject> right_identity = right->QueryInterface<IObject>();
  ASSERT_TRUE(left_identity && right_identity);
  EXPECT_EQ(left_identity.Get(), right_identity.Get());
  ObjPtr<IExtended> from_side = right->QueryInterface<IExtended>();
  ASSERT_EQ(from_side.Get(), object.Get());
  object.Release();
  left.Release();
  right.Release();
  left_identity.Release();
  right_identity.Release();
  EXPECT_FALSE(destroyed);
  from_side.Release();
  EXPECT_TRUE(destroyed);
}

TEST(RhiQueryInterface, ExtendedInterfacesExposeTheirDeclaredAncestors) {
  ObjPtr<IExtended> extended = new Extended;
  ObjPtr<ILeft> left = extended->QueryInterface<ILeft>();
  ASSERT_TRUE(left);
  ObjPtr<IExtended> roundtrip = left->QueryInterface<IExtended>();
  EXPECT_EQ(roundtrip.Get(), extended.Get());
  ObjPtr<IObject> identity = left->QueryInterface<IObject>();
  ASSERT_TRUE(identity);
  EXPECT_EQ(identity->QueryInterface<IRight>(), nullptr);
}

TEST(RhiQueryInterface, FailureClearsOutputAndDoesNotRetain) {
  bool destroyed = false;
  ObjPtr<Diamond> object = new Diamond(destroyed);
  void *output = object.Get();
  EXPECT_FALSE(object->QueryInterface(InterfaceID{}, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_FALSE(object->QueryInterface(uuid<ILeft>(), nullptr));
  EXPECT_EQ(object->QueryInterface<IBuffer>(), nullptr);
  EXPECT_TRUE(object->IsUniquelyOwned());
  object.Release();
  EXPECT_TRUE(destroyed);
}
}  // namespace query_test
