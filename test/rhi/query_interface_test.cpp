#include <gtest/gtest.h>

#include "grassland/rhi/backend/backend_util.h"

namespace query_test {
class ILeft : public grassland::rhi::IObject {};

class IRight : public grassland::rhi::IObject {};

class IExtended : public ILeft {};
class Diamond;
}  // namespace query_test

LONGMARCH_RHI_UUID(query_test::ILeft, "523d719e-fb75-41e8-addc-fc6abcf4575f")
LONGMARCH_RHI_UUID(query_test::IRight, "0ae16d45-31b1-4981-bba0-90a7df182c7e")
LONGMARCH_RHI_UUID(query_test::Diamond, "31feaf71-6632-45be-94a3-d3c82599c9d2")

LONGMARCH_RHI_UUID(query_test::IExtended, "beff92a1-db2c-47cb-80d9-0abff454b583")

namespace query_test {
using namespace grassland::rhi;
using namespace grassland::rhi::backend;

class Diamond final : public Object<ILeft>, public IRight {
 public:
  LONGMARCH_RHI_QUERY_MULTIPLE(Diamond, Object<ILeft>, IRight)

  explicit Diamond(bool &destroyed) : destroyed_(destroyed) {
  }

  ~Diamond() override {
    destroyed_ = true;
  }

 private:
  bool &destroyed_;
};

constexpr auto parsed = ParseInterfaceID("523d719e-fb75-41e8-addc-fc6abcf4575f");
static_assert(parsed.has_value());
static_assert(parsed->high == 0x523d719efb7541e8ULL);
static_assert(parsed->low == 0xaddcfc6abcf4575fULL);

static_assert(!ParseInterfaceID("523d719e-fb75-41e8-addc-fc6abcf4575x"));

TEST(RhiQueryInterface, UUIDParsingIsStrictAndCaseIndependent) {
  EXPECT_EQ(ParseInterfaceID("523D719E-FB75-41E8-ADDC-FC6ABCF4575F"), parsed);
  for (const char *invalid : {"", "523d719efb7541e8addcfc6abcf4575f", "523d719e_fb75-41e8-addc-fc6abcf4575f",
                              "{523d719e-fb75-41e8-addc-fc6abcf4575f}", "523d719e-fb75-41e8-addc-fc6abcf4575f "})
    EXPECT_FALSE(ParseInterfaceID(invalid));
  EXPECT_EQ(*parsed, uuid<ILeft>());
  EXPECT_NE(uuid<ILeft>(), uuid<IRight>());
}

TEST(RhiQueryInterface, DiamondHasOneCounterAndCanonicalObjectIdentity) {
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
  ObjPtr<Diamond> from_side = right->QueryInterface<Diamond>();
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
  ObjPtr<IExtended> extended = new Object<IExtended, ILeft>;
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
