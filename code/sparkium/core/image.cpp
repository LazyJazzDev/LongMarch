#include "sparkium/core/image.h"

#include <cstring>
#include <stdexcept>

namespace sparkium {
Image::Image(DataUpdateTracker &tracker, std::unique_ptr<graphics::Image> image)
    : tracker_(&tracker),
      image_(std::move(image)) {
  if (!image_)
    throw std::invalid_argument("tracked images require a native image");
  tracker.Register(this);
}

Image::~Image() {
  if (tracker_)
    tracker_->Unregister(this);
}

void Image::Invalidate() {
  ++revision_;
  bytes_.clear();
  valid_.clear();
  dirty_.clear();
}

void Image::AddRegion(std::vector<Region> &regions, Region region) {
  auto area = [](Region r) { return uint64_t(r.right - r.left) * (r.bottom - r.top); };
  for (size_t i = 0; i < regions.size();) {
    auto other = regions[i];
    Region bounds{std::min(region.left, other.left), std::min(region.top, other.top),
                  std::max(region.right, other.right), std::max(region.bottom, other.bottom)};
    auto left = std::max(region.left, other.left), right = std::min(region.right, other.right);
    auto top = std::max(region.top, other.top), bottom = std::min(region.bottom, other.bottom);
    uint64_t overlap = left < right && top < bottom ? uint64_t(right - left) * (bottom - top) : 0;
    // Merge only exact rectangular unions: bounding boxes must not fill GPU-owned holes.
    if (area(bounds) - area(region) == area(other) - overlap) {
      if (bounds.left == other.left && bounds.top == other.top && bounds.right == other.right &&
          bounds.bottom == other.bottom)
        return;
      region = bounds;
      regions.erase(regions.begin() + i);
      i = 0;
    } else {
      ++i;
    }
  }
  regions.push_back(region);
}

bool Image::HasSnapshot(Region region) const {
  for (auto known : valid_)
    if (known.left <= region.left && known.top <= region.top && known.right >= region.right &&
        known.bottom >= region.bottom)
      return true;
  // Subtract known rectangles from the query, without scanning per-byte validity bits.
  std::vector<Region> remaining{region};
  for (auto known : valid_) {
    std::vector<Region> next;
    for (auto r : remaining) {
      auto left = std::max(r.left, known.left), right = std::min(r.right, known.right);
      auto top = std::max(r.top, known.top), bottom = std::min(r.bottom, known.bottom);
      if (left >= right || top >= bottom) {
        next.push_back(r);
        continue;
      }
      if (r.top < top)
        next.push_back({r.left, r.top, r.right, top});
      if (bottom < r.bottom)
        next.push_back({r.left, bottom, r.right, r.bottom});
      if (r.left < left)
        next.push_back({r.left, top, left, bottom});
      if (right < r.right)
        next.push_back({right, top, r.right, bottom});
    }
    if (next.empty())
      return true;
    remaining = std::move(next);
  }
  return false;
}

void Image::Update(const void *data) {
  Update(data, {0, 0}, Extent());
}

void Image::Update(const void *data, graphics::Offset2D offset, graphics::Extent2D extent) {
  if (!tracker_)
    throw std::logic_error("image is no longer registered with a DataUpdateTracker");
  auto full = Extent();
  if (offset.x < 0 || offset.y < 0 || uint64_t(offset.x) + extent.width > full.width ||
      uint64_t(offset.y) + extent.height > full.height)
    throw std::out_of_range("tracked image region");
  if (!extent.width || !extent.height)
    return;
  if (!data)
    throw std::invalid_argument("null image update data");
  Region region{uint32_t(offset.x), uint32_t(offset.y), uint32_t(offset.x) + extent.width,
                uint32_t(offset.y) + extent.height};
  size_t pixel = graphics::PixelSize(Format());
  size_t pitch = size_t(full.width) * pixel, row_bytes = size_t(extent.width) * pixel;
  size_t begin = size_t(region.top) * pitch + size_t(region.left) * pixel;
  const auto *source = static_cast<const uint8_t *>(data);
  if (HasSnapshot(region)) {
    bool unchanged = true;
    size_t compare_rows = row_bytes == pitch ? 1 : extent.height;
    size_t compare_bytes = row_bytes == pitch ? row_bytes * extent.height : row_bytes;
    for (size_t y = 0; y < compare_rows; ++y) {
      if (std::memcmp(bytes_.data() + begin + y * pitch, source + y * row_bytes, compare_bytes)) {
        unchanged = false;
        break;
      }
    }
    if (unchanged)
      return;
  }
  // Allocate the pixel snapshot lazily; validity and dirtiness are rectangle metadata.
  bytes_.resize(pitch * full.height);
  if (row_bytes == pitch) {
    std::memcpy(bytes_.data() + begin, source, row_bytes * extent.height);
  } else {
    for (size_t y = 0; y < extent.height; ++y)
      std::memcpy(bytes_.data() + begin + y * pitch, source + y * row_bytes, row_bytes);
  }
  AddRegion(valid_, region);
  AddRegion(dirty_, region);
  ++revision_;
}

void Image::RecordUploads(graphics::CommandContext &commands, size_t &copies, size_t &bytes) {
  size_t pixel = graphics::PixelSize(Format());
  size_t pitch = size_t(Extent().width) * pixel;
  std::vector<uint8_t> packed;
  for (auto region : dirty_) {
    size_t row_bytes = size_t(region.right - region.left) * pixel;
    size_t rows = region.bottom - region.top;
    auto *source = bytes_.data() + size_t(region.top) * pitch + size_t(region.left) * pixel;
    if (row_bytes != pitch && rows > 1) {
      // The graphics upload API consumes tightly packed pixels, one command per rectangle.
      packed.resize(row_bytes * rows);
      for (size_t y = 0; y < rows; ++y)
        std::memcpy(packed.data() + y * row_bytes, source + y * pitch, row_bytes);
      source = packed.data();
    }
    commands.CmdUploadImage(Get(), source, {int32_t(region.left), int32_t(region.top)},
                            {region.right - region.left, region.bottom - region.top});
    bytes += row_bytes * rows;
    ++copies;
  }
}
}  // namespace sparkium
