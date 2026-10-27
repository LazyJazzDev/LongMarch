#include <gtest/gtest.h>

#include <type_traits>

#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhix/rhix_core.h"

namespace grassland::rhi {
namespace {
struct Counts {
  int references{1};
  int retains{};
  int destroyed{};
};

// This implementation uses only the public interface, not any backend helper.
class PublicCore final : public rhi::ICore {
 public:
  grassland::rhi::BackendAPI GetBackendAPI() const override {
    return grassland::rhi::BACKEND_API_DEFAULT;
  }

  explicit PublicCore(Counts &counts) : counts_(counts) {
  }

  void AddRef() noexcept override {
    ++counts_.references;
    ++counts_.retains;
  }

  void Release() noexcept override {
    if (!--counts_.references)
      delete this;
  }

  std::vector<rhi::PhysicalDevice> GetPhysicalDevices() const override {
    return {};
  }

  rhi::IDevice *CreateDevice(int32_t) override {
    return nullptr;
  }

  rhi::IDevice *CreateDeviceByAutoSelection(rhi::DeviceFeatureRequirement) override {
    return nullptr;
  }

  ~PublicCore() override {
    ++counts_.destroyed;
  }

 private:
  Counts &counts_;
};

static_assert(sizeof(ObjPtr<rhi::ICore>) == sizeof(rhi::ICore *));
static_assert(std::is_convertible_v<rhi::ICore *, ObjPtr<rhi::ICore>>);
static_assert(!std::is_convertible_v<ObjPtr<rhi::ICore>, rhi::ICore *>);
static_assert(std::is_constructible_v<ObjPtr<rhi::IObject>, ObjPtr<rhi::ICore>>);
static_assert(!std::is_constructible_v<ObjPtr<rhi::ICore>, ObjPtr<rhi::IObject>>);

TEST(RhiObjPtr, RawPointerAdoptsLvalueRetainsAndRvalueTransfers) {
  Counts counts;
  ObjPtr<rhi::ICore> core = new PublicCore(counts);
  EXPECT_EQ(counts.retains, 0);
  ObjPtr<rhi::ICore> copied = core;
  EXPECT_EQ(counts.references, 2);
  EXPECT_EQ(counts.retains, 1);
  ObjPtr<rhi::IObject> moved = std::move(copied);
  EXPECT_FALSE(copied);
  EXPECT_EQ(counts.references, 2);
  auto retained = ObjPtr<rhi::ICore>::Retain(core.Get());
  EXPECT_EQ(counts.references, 3);
  auto &self = core;
  core = self;
  core = std::move(self);
  EXPECT_TRUE(core);
  EXPECT_EQ(counts.references, 3);
  retained.Release();
  auto raw = core.Detach();
  EXPECT_FALSE(core);
  EXPECT_EQ(counts.references, 2);
  ObjPtr<rhi::ICore> adopted = raw;
  moved.Release();
  EXPECT_EQ(counts.destroyed, 0);
  adopted.Release();
  adopted.Release();
  EXPECT_EQ(counts.destroyed, 1);
}

TEST(RhiObjPtr, RawReplacementAndNullDoNotRetain) {
  Counts first, second;
  ObjPtr<rhi::ICore> core = new PublicCore(first);
  core = new PublicCore(second);
  EXPECT_EQ(first.destroyed, 1);
  EXPECT_EQ(second.references, 1);
  EXPECT_EQ(second.retains, 0);
  core = nullptr;
  EXPECT_EQ(second.destroyed, 1);
  EXPECT_FALSE(core);
  EXPECT_FALSE(ObjPtr<rhi::ICore>::Retain(nullptr));
}

TEST(RhixWrappers, DotSyntaxUsesOnlyPublicInterfaceAndAdoptsRawPointer) {
  Counts counts;
  rhix::Core core = new PublicCore(counts);
  EXPECT_EQ(counts.retains, 0);
  EXPECT_TRUE(core.GetPhysicalDevices().empty());
  EXPECT_FALSE(core.CreateDevice(0));
  rhix::Core copy = core;
  rhix::Core moved = std::move(copy);
  EXPECT_FALSE(copy);
  core.Release();
  EXPECT_EQ(counts.destroyed, 0);
  moved.Release();
  EXPECT_EQ(counts.destroyed, 1);
}
}  // namespace
}  // namespace grassland::rhi
