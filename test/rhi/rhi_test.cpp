#include "grassland/rhi/rhi.h"

#include <gtest/gtest.h>

#include <utility>

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi {
namespace {

PhysicalDevice MakePhysicalDevice(PhysicalDeviceType type, uint64_t gib, bool pipeline, bool query, bool geometry) {
  PhysicalDevice info{};
  info.name = "Test device";
  info.type = type;
  if (type == PHYSICAL_DEVICE_TYPE_DEDICATED)
    info.dedicated_memory = gib << 30;
  else
    info.unified_memory = gib << 30;
  info.ray_tracing_pipeline_feature = pipeline;
  info.ray_query_feature = query;
  info.geometry_shader_feature = geometry;
  return info;
}

struct Calls {
  int selected{-1};
  int creates{};
  int core_destructions{};
  int device_destructions{};
  bool device_destroyed_before_core{};
};

class TestDevice final : public backend::IDevice {
 public:
  TestDevice(PhysicalDevice info, const backend::ObjectPtr<backend::ICore> &core, Calls &calls)
      : IDevice(std::move(info), core),
        calls_(calls) {
  }

  ~TestDevice() override {
    ++calls_.device_destructions;
    calls_.device_destroyed_before_core = calls_.core_destructions == 0;
  }

 private:
  Calls &calls_;
};

class TestCore final : public backend::ICore {
 public:
  TestCore(std::vector<PhysicalDevice> devices, Calls &calls, bool fail = false)
      : devices_(std::move(devices)),
        calls_(calls),
        fail_(fail) {
  }

  ~TestCore() override {
    ++calls_.core_destructions;
  }

  std::vector<PhysicalDevice> GetPhysicalDevices() const override {
    return devices_;
  }

  std::unique_ptr<backend::IDevice> CreateDevice(int32_t id,
                                                 const backend::ObjectPtr<backend::ICore> &owner) const override {
    calls_.selected = id;
    ++calls_.creates;
    if (fail_)
      return {};
    return std::make_unique<TestDevice>(devices_.at(id), owner, calls_);
  }

 private:
  std::vector<PhysicalDevice> devices_;
  Calls &calls_;
  bool fail_;
};

TEST(RhiSelection, RequirementsAreIndependentFilters) {
  const std::vector<PhysicalDevice> devices = {
      MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 24, false, false, false),
      MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_INTEGRATED, 8, true, false, false),
      MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_INTEGRATED, 8, false, true, false),
      MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_INTEGRATED, 8, false, false, true)};
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {}), 0);
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {true, false, false}), 1);
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {false, true, false}), 2);
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {false, false, true}), 3);
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {true, true, true}), -1);
  EXPECT_EQ(backend::SelectPhysicalDevice({}, {}), -1);
}

TEST(RhiSelection, RankingPrefersHardwareThenMemoryAndFeaturesWithStableTies) {
  std::vector<PhysicalDevice> devices = {MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_SOFTWARE, 256, true, true, true),
                                         MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_INTEGRATED, 8, false, false, false)};
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {}), 1);
  devices.push_back(MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 4, false, false, false));
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {}), 2);
  devices.push_back(MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 8, false, false, false));
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {}), 3);
  devices.push_back(MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 8, true, true, true));
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {}), 4);
  devices.push_back(devices.back());
  EXPECT_EQ(backend::SelectPhysicalDevice(devices, {}), 4);
}

TEST(RhiCore, AutoSelectionForwardsOnlyTheSelectedIdAndPreservesAllFeatures) {
  Calls calls;
  auto core = backend::Access::Adopt<Core>(std::make_unique<TestCore>(
      std::vector<PhysicalDevice>{MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 24, false, false, true),
                                  MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_INTEGRATED, 8, true, true, true)},
      calls));
  auto device = core.CreateDeviceByAutoSelection({false, true, false});
  ASSERT_TRUE(device);
  EXPECT_EQ(calls.selected, 1);
  EXPECT_EQ(calls.creates, 1);
  EXPECT_TRUE(device.SupportRayTracingPipeline());
  EXPECT_TRUE(device.SupportRayQuery());
  EXPECT_TRUE(device.SupportGeometryShader());
  auto explicit_device = core.CreateDevice(0);
  ASSERT_TRUE(explicit_device);
  EXPECT_EQ(calls.selected, 0);
  EXPECT_FALSE(explicit_device.SupportRayQuery());
  EXPECT_TRUE(explicit_device.SupportGeometryShader());
}

TEST(RhiCore, InvalidIdsAndUnmatchedRequirementsNeverReachTheBackend) {
  Calls calls;
  auto core = backend::Access::Adopt<Core>(std::make_unique<TestCore>(
      std::vector<PhysicalDevice>{MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 8, false, false, true)}, calls));
  EXPECT_FALSE(core.CreateDevice(-1));
  EXPECT_FALSE(core.CreateDevice(1));
  EXPECT_FALSE(core.CreateDeviceByAutoSelection({true, false, false}));
  EXPECT_EQ(calls.creates, 0);
}

TEST(RhiCore, CreationFailureIsEmptyAndDoesNotSilentlyChooseAnotherDevice) {
  Calls calls;
  auto info = MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 8, true, true, true);
  auto core =
      backend::Access::Adopt<Core>(std::make_unique<TestCore>(std::vector<PhysicalDevice>{info, info}, calls, true));
  EXPECT_FALSE(core.CreateDeviceByAutoSelection());
  EXPECT_EQ(calls.selected, 0);
  EXPECT_EQ(calls.creates, 1);
}

TEST(RhiOwnership, DeviceKeepsCoreAliveAcrossCopyMoveAndRelease) {
  Calls calls;
  auto core = backend::Access::Adopt<Core>(std::make_unique<TestCore>(
      std::vector<PhysicalDevice>{MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 8, true, true, true)}, calls));
  Core copy = core;
  Core moved = std::move(copy);
  EXPECT_FALSE(copy);
  auto device = moved.CreateDevice(0);
  core = {};
  moved = {};
  EXPECT_EQ(calls.core_destructions, 0);
  Device shared = device;
  device = {};
  EXPECT_EQ(shared.Name(), "Test device");
  EXPECT_TRUE(shared.SupportRayQuery());
  EXPECT_EQ(calls.device_destructions, 0);
  shared = {};
  EXPECT_EQ(calls.device_destructions, 1);
  EXPECT_EQ(calls.core_destructions, 1);
  EXPECT_TRUE(calls.device_destroyed_before_core);
}

TEST(RhiCore, EmptyHandlesAndUnavailableBackendFailSafely) {
  Core core;
  EXPECT_TRUE(core.GetPhysicalDevices().empty());
  EXPECT_FALSE(core.CreateDevice(0));
  EXPECT_FALSE(core.CreateDeviceByAutoSelection());
  Device device;
  EXPECT_TRUE(device.Name().empty());
  EXPECT_FALSE(device.SupportGeometryShader());
  EXPECT_FALSE(device.SupportRayQuery());
  EXPECT_FALSE(device.SupportRayTracingPipeline());
  EXPECT_FALSE(CreateCore(BACKEND_API_METAL));
}

class RhiNative : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiNative, EnumerateCreateEveryDeviceAndSelectWithAllRequirementCombinations) {
  auto core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  const auto devices = core.GetPhysicalDevices();
  if (devices.empty())
    GTEST_SKIP() << "No compatible devices on this machine";
  EXPECT_EQ(devices.size(), core.GetPhysicalDevices().size());
  EXPECT_FALSE(core.CreateDevice(-1));
  EXPECT_FALSE(core.CreateDevice(static_cast<int32_t>(devices.size())));
  for (size_t id = 0; id < devices.size(); ++id) {
    auto device = core.CreateDevice(static_cast<int32_t>(id));
    ASSERT_TRUE(device) << devices[id].name;
    EXPECT_EQ(device.Name(), devices[id].name);
    EXPECT_EQ(device.SupportRayTracingPipeline(), devices[id].ray_tracing_pipeline_feature);
    EXPECT_EQ(device.SupportRayQuery(), devices[id].ray_query_feature);
    EXPECT_EQ(device.SupportGeometryShader(), devices[id].geometry_shader_feature);
  }
  for (unsigned mask = 0; mask < 8; ++mask) {
    DeviceFeatureRequirement requirement{bool(mask & 1), bool(mask & 2), bool(mask & 4)};
    const auto id = backend::SelectPhysicalDevice(devices, requirement);
    auto device = core.CreateDeviceByAutoSelection(requirement);
    ASSERT_EQ(static_cast<bool>(device), id >= 0);
    if (id >= 0) {
      EXPECT_EQ(device.Name(), devices[id].name);
      EXPECT_EQ(device.SupportRayTracingPipeline(), devices[id].ray_tracing_pipeline_feature);
      EXPECT_EQ(device.SupportRayQuery(), devices[id].ray_query_feature);
      EXPECT_EQ(device.SupportGeometryShader(), devices[id].geometry_shader_feature);
    }
  }
  auto device = core.CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  auto name = device.Name();
  core = {};
  EXPECT_EQ(device.Name(), name);
  device = {};
}

std::vector<BackendAPI> NativeBackends() {
  return {
#if defined(LONGMARCH_D3D12_ENABLED)
      BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
      BACKEND_API_VULKAN,
#endif
  };
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiNative, testing::ValuesIn(NativeBackends()));

}  // namespace
}  // namespace grassland::rhi
