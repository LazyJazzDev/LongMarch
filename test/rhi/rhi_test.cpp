#include <gtest/gtest.h>

#include <algorithm>
#include <type_traits>
#include <utility>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_shader.h"
#include "grassland/rhi/backend/backend_shader_impl.h"
#include "grassland/rhi/rhix/rhix.h"

using grassland::rhi::backend::MatchInterfaces;

namespace grassland::rhix {
using namespace rhi;
namespace backend = rhi::backend;
using namespace backend;

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

class TestDevice final : public backend::Object<backend::IBackendDevice> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IDevice, backend::IBackendDevice, rhi::IDevice)

  bool WaitNativeEvents(const std::vector<Event> &, bool) const override {
    ADD_FAILURE();
    return false;
  }

  void InterruptCollector() override {
    ADD_FAILURE();
  }

  void WaitForProgress(const std::vector<Event> &) override {
    ADD_FAILURE();
  }

  TestDevice(PhysicalDevice info, backend::IBackendCore *core, Calls &calls) : calls_(calls) {
    InitializeDevice(std::move(info), core, BACKEND_API_DEFAULT);
  }

  ~TestDevice() override {
    ++calls_.device_destructions;
    calls_.device_destroyed_before_core = calls_.core_destructions == 0;
    DestroyDevice();
  }

 private:
  Calls &calls_;

 public:
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &) override {
    ReportError(-1, "Acceleration structures are unavailable for this backend.");
    return nullptr;
  }

  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &) override {
    ReportError(-1, "Acceleration structures are unavailable for this backend.");
    return nullptr;
  }

  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &) override {
    ReportError(-1, "Ray tracing pipelines are unavailable for this backend.");
    return nullptr;
  }

 private:
  void InitializeDevice(PhysicalDevice info, IBackendCore *core, BackendAPI api) {
    api_ = api;
    core_ = core;
    info_ = std::move(info);
  }

 public:
  rhi::IImage *CreateImage(const ImageDesc &) override {
    ReportError(-1, "Independent images are not implemented for this backend.");
    return nullptr;
  }

  rhi::IBuffer *CreateBuffer(const BufferDesc &) override {
    ReportError(-1, "Buffers are not implemented for this backend.");
    return nullptr;
  }

  rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &) override {
    ReportError(-1, "Compute programs are not implemented for this backend.");
    return nullptr;
  }

  rhi::IProgram *CreateProgram(const ProgramDesc &) override {
    ReportError(-1, "Graphics programs are not implemented for this backend.");
    return nullptr;
  }

  rhi::IShader *CreateShader(const ShaderDesc &desc) override {
    if (desc.filename.empty() || desc.entry_point.empty() || desc.stage < SHADER_STAGE_VERTEX ||
        desc.stage > SHADER_STAGE_CALLABLE)
      return nullptr;
    return IBackendShader::Create(desc, api_);
  }

  ISampler *CreateSampler(const SamplerDesc &) override {
    ReportError(-1, "Samplers are not implemented for this backend.");
    return nullptr;
  }

  IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &) override {
    ReportError(-1, "Binding groups are not implemented for this backend.");
    return nullptr;
  }

 private:
  void DestroyDevice() {
    ShutdownQueues();
  }

 public:
  std::string Name() const override {
    return info_.name;
  }

  bool SupportRayTracingPipeline() const override {
    return info_.ray_tracing_pipeline_feature;
  }

  bool SupportRayQuery() const override {
    return info_.ray_query_feature;
  }

  bool SupportGeometryShader() const override {
    return info_.geometry_shader_feature;
  }

  rhi::IWindow *CreateGLFWWindow(const WindowDesc &) override {
    ReportError(-1, "Window creation is unavailable.");
    return nullptr;
  }

  rhi::IQueue *GetQueue(QueueType type) override {
    if (type != QUEUE_TYPE_MAIN && type != QUEUE_TYPE_ASSIST)
      return nullptr;
    return ObjectPtr<IBackendQueue>::Retain(queues_[type].Get()).Detach();
  }

  const PhysicalDevice &GetPhysicalDevice() const noexcept override {
    return info_;
  }

 private:
  IBackendCore *OwnerCore() const {
    return core_;
  }

  void StartCollector() {
    collector_ = std::thread([this] { Collect(); });
  }

 public:
  bool WaitEvents(const std::vector<Event> &events, bool wait_all) const override {
    if (events.empty())
      return wait_all;
    std::vector<Event> unique;
    unique.reserve(events.size());
    for (const auto &event : events) {
      auto timeline = rhi::QueryInterfacePtr<IBackendTimeline>(event.timeline).Get();
      if (!timeline || timeline->Owner() != this) {
        ReportError(-1, "Host waits require timelines from this device.");
        return false;
      }
      auto found = std::find_if(unique.begin(), unique.end(),
                                [&](const Event &other) { return other.timeline == event.timeline; });
      if (found == unique.end())
        unique.push_back(event);
      else
        found->value = wait_all ? std::max(found->value, event.value) : std::min(found->value, event.value);
    }
    return WaitNativeEvents(unique, wait_all);
  }

 private:
  void Collect() {
    std::vector<Event> next;
    next.reserve(queues_.size());
    for (;;) {
      next.clear();
      bool empty = true;
      for (const auto &queue : queues_) {
        if (!queue)
          continue;
        uint64_t completed = 0;
        const bool timeline_valid = queue->GetTimeline()->GetCompletedValue(completed);
        if (timeline_valid && completed != UINT64_MAX)
          next.push_back({queue->GetTimeline(), completed + 1});
        if (!queue->CollectCompleted(completed, timeline_valid))
          empty = false;
      }
      if (stopping_ && empty)
        return;
      WaitForProgress(next);
    }
  }

  void ShutdownQueues() {
    if (!collector_.joinable())
      return;
    stopping_ = true;
    InterruptCollector();
    collector_.join();
  }

  std::array<ObjectPtr<IBackendQueue>, 2> queues_{};
  std::atomic<bool> stopping_{};
  BackendAPI api_{};
  IBackendCore *core_{};
  PhysicalDevice info_{};
  std::thread collector_{};
};

class TestCore final : public backend::Object<backend::IBackendCore> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::ICore, backend::IBackendCore, rhi::ICore)

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

  IDevice *CreateDevice(int32_t id) override {
#if !defined(NDEBUG)
    if (id < 0 || static_cast<size_t>(id) >= devices_.size())
      return nullptr;
#endif
    calls_.selected = id;
    ++calls_.creates;
    if (fail_)
      return {};
    return new TestDevice(devices_.at(id), this, calls_);
  }

 private:
  std::vector<PhysicalDevice> devices_;
  Calls &calls_;
  bool fail_;

 public:
  rhi::IDevice *CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {}) override {
    return backend::IBackendCore::CommonCreateDeviceByAutoSelection(requirement);
  }
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
  auto core = Core(new TestCore(
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
  auto core = Core(new TestCore(
      std::vector<PhysicalDevice>{MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 8, false, false, true)}, calls));
#if !defined(NDEBUG)
  EXPECT_FALSE(core.CreateDevice(-1));
  EXPECT_FALSE(core.CreateDevice(1));
#endif
  EXPECT_FALSE(core.CreateDeviceByAutoSelection({true, false, false}));
  EXPECT_EQ(calls.creates, 0);
}

TEST(RhiCore, CreationFailureIsEmptyAndDoesNotSilentlyChooseAnotherDevice) {
  Calls calls;
  auto info = MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 8, true, true, true);
  auto core = Core(new TestCore(std::vector<PhysicalDevice>{info, info}, calls, true));
  EXPECT_FALSE(core.CreateDeviceByAutoSelection());
  EXPECT_EQ(calls.selected, 0);
  EXPECT_EQ(calls.creates, 1);
}

TEST(RhiOwnership, CallerReleasesAllChildCopiesBeforeParent) {
  static_assert(std::is_base_of_v<ObjectContainer, Core>);
  static_assert(std::is_base_of_v<ObjectContainer, Device>);
  static_assert(std::is_base_of_v<ObjectContainer, Window>);
  static_assert(std::is_base_of_v<ObjectContainer, Image>);
  static_assert(std::is_base_of_v<ObjectContainer, Queue>);
  static_assert(std::is_base_of_v<ObjectContainer, SubmissionToken>);
  static_assert(std::is_base_of_v<ObjectContainer, CommandContext>);
  static_assert(sizeof(Core) == sizeof(ObjPtr<rhi::IObject>));
  static_assert(sizeof(Image) == sizeof(ObjPtr<rhi::IObject>));
  Calls calls;
  auto core = Core(new TestCore(
      std::vector<PhysicalDevice>{MakePhysicalDevice(PHYSICAL_DEVICE_TYPE_DEDICATED, 8, true, true, true)}, calls));
  Core copy = core;
  Core moved = std::move(copy);
  EXPECT_FALSE(copy);
  auto device = moved.CreateDevice(0);
  core = {};
  EXPECT_TRUE((*static_cast<TestCore *>((moved).Get())).IsUniquelyOwned());
  EXPECT_EQ(calls.core_destructions, 0);
  Device shared = device;
  ObjectContainer &container = device;
  container.Release();
  container.Release();  // Empty release is safe and does not affect another copy.
  EXPECT_FALSE(device);
  EXPECT_EQ(shared.Name(), "Test device");
  EXPECT_TRUE(shared.SupportRayQuery());
  EXPECT_EQ(calls.device_destructions, 0);
  shared.Release();
  EXPECT_EQ(calls.core_destructions, 0);
  moved.Release();
  EXPECT_EQ(calls.device_destructions, 1);
  EXPECT_EQ(calls.core_destructions, 1);
  EXPECT_TRUE(calls.device_destroyed_before_core);
}

TEST(RhiCore, EmptyHandlesAndUnavailableBackendFailSafely) {
#if !defined(NDEBUG)
  Core core;
  EXPECT_TRUE(core.GetPhysicalDevices().empty());
  EXPECT_FALSE(core.CreateDevice(0));
  EXPECT_FALSE(core.CreateDeviceByAutoSelection());
  Device device;
  EXPECT_TRUE(device.Name().empty());
  EXPECT_FALSE(device.SupportGeometryShader());
  EXPECT_FALSE(device.SupportRayQuery());
  EXPECT_FALSE(device.SupportRayTracingPipeline());
#endif
  EXPECT_FALSE(rhix::CreateCore(static_cast<BackendAPI>(-1)));
}

class RhiNative : public testing::TestWithParam<BackendAPI> {};

TEST_P(RhiNative, EnumerateCreateEveryDeviceAndSelectWithAllRequirementCombinations) {
  auto core = rhix::CreateCore(GetParam());
  ASSERT_TRUE(core);
  const auto devices = core.GetPhysicalDevices();
  if (devices.empty())
    GTEST_SKIP() << "No compatible devices on this machine";
  EXPECT_EQ(devices.size(), core.GetPhysicalDevices().size());
#if !defined(NDEBUG)
  EXPECT_FALSE(core.CreateDevice(-1));
  EXPECT_FALSE(core.CreateDevice(static_cast<int32_t>(devices.size())));
#endif
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
  EXPECT_EQ(device.Name(), name);
  device.Release();
  core.Release();
}

std::vector<BackendAPI> NativeBackends() {
  return {
#if defined(LONGMARCH_D3D12_ENABLED)
      BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
      BACKEND_API_VULKAN,
#endif
#if defined(LONGMARCH_METAL_ENABLED)
      BACKEND_API_METAL,
#endif
  };
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiNative, testing::ValuesIn(NativeBackends()));

}  // namespace
}  // namespace grassland::rhix
