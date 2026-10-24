#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstring>
#include <thread>

#include "grassland/rhi/rhix/rhix_download_token.h"

#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/rhi/backend/metal/metal_device.h"
#endif
#include "grassland/rhi/rhi.h"

namespace grassland::rhi {
class RhiDownload : public testing::TestWithParam<BackendAPI> {
 protected:
  ICore *CreateCore(BackendAPI api) {
    return grassland::rhi::CreateCore(api, {true});
  }
};

TEST_P(RhiDownload, BufferRangeSnapshotsAndRetainedLeases) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IBuffer> buffer = device->CreateBuffer({256});
  ASSERT_TRUE(buffer);
  std::array<uint32_t, 64> source{};
  for (uint32_t i = 0; i < source.size(); ++i)
    source[i] = i * 137 + 19;
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, source.data(), sizeof(source)));
  ObjPtr<IDownloadToken> first = commands->CmdDownloadBuffer(buffer.Get(), 12, 100);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->GetSize(), 100);
  ObjPtr<IDownloadToken> unaligned = commands->CmdDownloadBuffer(buffer.Get(), 7, 19);
  ASSERT_TRUE(unaligned);
  std::array<uint32_t, 25> output{};
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done);
  commands.Release();
  ASSERT_TRUE(done->WaitHost());
  std::array<uint8_t, 19> unaligned_bytes{};
  ASSERT_TRUE(unaligned->CopyTo(unaligned_bytes.data(), unaligned_bytes.size()));
  EXPECT_EQ(std::memcmp(unaligned_bytes.data(), reinterpret_cast<const uint8_t *>(source.data()) + 7, 19), 0);
  ASSERT_TRUE(first->CopyTo(output.data(), sizeof(output)));
  EXPECT_EQ(std::memcmp(output.data(), source.data() + 3, sizeof(output)), 0);
  output.fill(0);
  EXPECT_FALSE(first->CopyTo(output.data(), sizeof(output) - 1));
  EXPECT_EQ(output[0], 0);
  EXPECT_FALSE(first->CopyTo(nullptr, sizeof(output)));
  // A completed token holds its snapshot across later uploads/downloads.
  for (int i = 0; i < 8; ++i) {
    commands = device->CreateCommandContext();
    std::array<uint32_t, 64> replacement{};
    replacement.fill(900 + i);
    ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, replacement.data(), sizeof(replacement)));
    ObjPtr<IDownloadToken> next = commands->CmdDownloadBuffer(buffer.Get(), 0, 100);
    ASSERT_TRUE(next);
    if (i % 2)
      next.Release();  // Early release must not recycle in-flight storage.
    done = queue->Submit(commands.Get());
    ASSERT_TRUE(done && done->WaitHost());
    if (next) {
      ASSERT_TRUE(next->CopyTo(output.data(), sizeof(output)));
      EXPECT_EQ(output[0], 900 + i);
    }
    ASSERT_TRUE(first->CopyTo(output.data(), sizeof(output)));
    EXPECT_EQ(std::memcmp(output.data(), source.data() + 3, sizeof(output)), 0);
  }
}

TEST_P(RhiDownload, AbandonedAndInvalidDownloads) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IBuffer> buffer = device->CreateBuffer({64});
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ObjPtr<IDownloadToken> token = commands->CmdDownloadBuffer(buffer.Get(), 0, 64);
  ASSERT_TRUE(token);
  commands.Release();
  // Abandonment is owned by the caller; release the unused download explicitly.
  token.Release();
  for (auto range : {std::pair<uint64_t, uint64_t>{64, 1}, {0, 0}, {UINT64_MAX, 16}, {1, 64}}) {
    commands = device->CreateCommandContext();
    EXPECT_EQ(commands->CmdDownloadBuffer(buffer.Get(), range.first, range.second), nullptr);
  }
}

TEST_P(RhiDownload, ImageSubrectangleIsTightlyPackedAndUnconverted) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  for (auto format : {IMAGE_FORMAT_RGBA8_UNORM, IMAGE_FORMAT_BGRA8_UNORM, IMAGE_FORMAT_RGBA16_FLOAT,
                      IMAGE_FORMAT_RGB10A2_UNORM, IMAGE_FORMAT_BGR10A2_UNORM}) {
    if (format == IMAGE_FORMAT_BGR10A2_UNORM && GetParam() == BACKEND_API_D3D12)
      continue;
    SCOPED_TRACE(ImageFormatName(format));
    const uint32_t pixel = ImageFormatSize(format), pitch = 11 * pixel + 16;
    std::vector<uint8_t> source(pitch * 7);
    for (size_t i = 0; i < source.size(); ++i)
      source[i] = uint8_t(i * 13 + 17);
    ObjPtr<IImage> image = device->CreateImage({11, 7, format});
    ASSERT_TRUE(image);
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ASSERT_TRUE(commands->CmdUploadImage(image.Get(), source.data(), pitch));
    ObjPtr<IDownloadToken> region = commands->CmdDownloadImage(image.Get(), {2, 1, 5, 3});
    ASSERT_TRUE(region);
    ObjPtr<IDownloadToken> edge = commands->CmdDownloadImage(image.Get(), {9, 5});
    ASSERT_TRUE(edge);
    ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
    ASSERT_TRUE(done);
    image.Release();
    commands.Release();
    ASSERT_TRUE(done->WaitHost());
    EXPECT_EQ(region->GetSize(), 5 * 3 * pixel);
    std::vector<uint8_t> actual(region->GetSize() + 8, 0xcd);
    ASSERT_TRUE(region->CopyTo(actual.data(), region->GetSize()));
    for (int y = 0; y < 3; ++y)
      EXPECT_EQ(std::memcmp(actual.data() + y * 5 * pixel, source.data() + (y + 1) * pitch + 2 * pixel, 5 * pixel), 0);
    EXPECT_EQ(actual.back(), 0xcd);
    EXPECT_EQ(edge->GetSize(), 2 * 2 * pixel);
    ASSERT_TRUE(edge->CopyTo(actual.data(), actual.size()));
    for (int y = 0; y < 2; ++y)
      EXPECT_EQ(std::memcmp(actual.data() + y * 2 * pixel, source.data() + (y + 5) * pitch + 9 * pixel, 2 * pixel), 0);
  }
}

TEST_P(RhiDownload, DepthSubrectangleIsTightlyPacked) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  const auto format = IMAGE_FORMAT_D32_FLOAT;
  ObjPtr<IImage> image = device->CreateImage({13, 7, format});
  ASSERT_TRUE(image);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  RenderingInfo info;
  info.depth_stencil.image = image.Get();
  info.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  info.depth_stencil.clear_depth = 0.25f;
  ASSERT_TRUE(commands->CmdBeginRendering(info));
  ASSERT_TRUE(commands->CmdEndRendering());
  ObjPtr<IDownloadToken> token = commands->CmdDownloadImage(image.Get(), {3, 2, 5, 3});
  ASSERT_TRUE(token);
  ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
  ASSERT_TRUE(done && done->WaitHost());
  ASSERT_EQ(token->GetSize(), 60);
  std::array<uint32_t, 15> values{};
  ASSERT_TRUE(token->CopyTo(values.data(), sizeof(values)));
  for (auto value : values) {
    float depth;
    std::memcpy(&depth, &value, 4);
    EXPECT_FLOAT_EQ(depth, 0.25f);
  }
}

TEST_P(RhiDownload, RejectedSubmissionLeavesDownloadUnassociated) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IBuffer> buffer = device->CreateBuffer({64});
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  ObjPtr<IDownloadToken> token = commands->CmdDownloadBuffer(buffer.Get(), 0, 64);
  ASSERT_TRUE(token);
  EXPECT_EQ(commands->CmdDownloadBuffer(buffer.Get(), 64, 1), nullptr);
  ObjPtr<ISubmissionToken> rejected = queue->Submit(commands.Get());
  EXPECT_FALSE(rejected);
  EXPECT_EQ(token->GetStatus(), SUBMISSION_STATUS_PENDING);
  EXPECT_FALSE(token->WaitHost());
}

TEST_P(RhiDownload, WaitHostSurvivesSubmissionHandleReleaseAndSupportsMultipleWaiters) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IBuffer> buffer = device->CreateBuffer({64});
  ASSERT_TRUE(buffer);
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  std::array<uint32_t, 16> source{};
  source.fill(0x12345678);
  ASSERT_TRUE(commands->CmdUploadBuffer(buffer.Get(), 0, source.data(), sizeof(source)));
  ObjPtr<IDownloadToken> token = commands->CmdDownloadBuffer(buffer.Get(), 0, sizeof(source));
  ASSERT_TRUE(token);
  ObjPtr<IDownloadToken> other = commands->CmdDownloadBuffer(buffer.Get(), 4, 8);
  ASSERT_TRUE(other);
  ObjPtr<ISubmissionToken> submitted = queue->Submit(commands.Get());
  ASSERT_TRUE(submitted);
  submitted.Release();
  commands.Release();
  buffer.Release();
  bool first = false, second = false;
  std::thread a([&] { first = token->WaitHost(); });
  std::thread b([&] { second = token->WaitHost(); });
  a.join();
  b.join();
  EXPECT_TRUE(first);
  EXPECT_TRUE(second);
  EXPECT_TRUE(other->WaitHost());
  grassland::rhix::DownloadToken wrapped(token.Detach());
  EXPECT_TRUE(wrapped.WaitHost());
  std::array<uint32_t, 16> output{};
  ASSERT_TRUE(wrapped.CopyTo(output.data(), sizeof(output)));
  EXPECT_EQ(output, source);
}

TEST_P(RhiDownload, TemporalPoolRetainsTokenLeaseAndShrinksExpiredDemand) {
  ObjPtr<ICore> core = CreateCore(GetParam());
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IBuffer> buffer = device->CreateBuffer({64});
  ASSERT_TRUE(buffer);
  {
    ObjPtr<ICommandContext> upload = device->CreateCommandContext();
    std::array<uint32_t, 16> data{};
    ASSERT_TRUE(upload->CmdUploadBuffer(buffer.Get(), 0, data.data(), sizeof(data)));
    ObjPtr<ISubmissionToken> submitted = queue->Submit(upload.Get());
    upload.Release();
    ASSERT_TRUE(submitted && submitted->WaitHost());
  }
  auto check = [&](auto *owner) {
    auto &pool = owner->DownloadPool();
    pool.Clear();
    EXPECT_EQ(&pool == &owner->UploadPool(), GetParam() == BACKEND_API_METAL);
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    ObjPtr<IDownloadToken> token = commands->CmdDownloadBuffer(buffer.Get(), 0, 64);
    ASSERT_TRUE(token);
    EXPECT_EQ(pool.GetBufferCounts(64).second, 1u);
    ObjPtr<ISubmissionToken> submitted = queue->Submit(commands.Get());
    ASSERT_TRUE(submitted);
    commands.Release();
    submitted.Release();
    ASSERT_TRUE(token->WaitHost());
    EXPECT_EQ(pool.GetBufferCounts(64).second, 1u);
    auto other = pool.Acquire(64);
    ASSERT_TRUE(other);
    EXPECT_EQ(pool.GetBufferCounts(64).first, 2u);
    EXPECT_EQ(pool.GetBufferCounts(64).second, 2u);
    token.Release();
    EXPECT_EQ(pool.GetBufferCounts(64).second, 1u);
    auto reused = pool.Acquire(64);
    ASSERT_TRUE(reused);
    EXPECT_EQ(pool.GetBufferCounts(64).first, 2u);
    other.Reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    reused.Reset();
    EXPECT_EQ(pool.GetBufferCounts(64).first, 0u);
    EXPECT_EQ(pool.GetBufferCounts(64).second, 0u);
    if (GetParam() == BACKEND_API_METAL) {
      auto upload = owner->UploadPool().Acquire(64);
      ASSERT_TRUE(upload);
      auto native = upload->NativeBuffer();
      upload.Reset();
      auto download = pool.Acquire(64);
      ASSERT_TRUE(download);
      EXPECT_EQ(download->NativeBuffer(), native);
    }
  };
#if defined(LONGMARCH_D3D12_ENABLED)
  if (GetParam() == BACKEND_API_D3D12)
    check(static_cast<backend::d3d12::D3D12Device *>(device.Get()));
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (GetParam() == BACKEND_API_VULKAN)
    check(static_cast<backend::vulkan::VulkanDevice *>(device.Get()));
#endif
#if defined(LONGMARCH_METAL_ENABLED)
  if (GetParam() == BACKEND_API_METAL)
    check(static_cast<backend::metal::MetalDevice *>(device.Get()));
#endif
}

const BackendAPI download_backends[] = {
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
INSTANTIATE_TEST_SUITE_P(Backends, RhiDownload, testing::ValuesIn(download_backends));
}  // namespace grassland::rhi
