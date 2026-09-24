#pragma once

#include <CoreGraphics/CoreGraphics.h>

#include <algorithm>
#include <cstring>
#include <vector>

// Tag the actual exposed RGB range: untagged extended-linear CGImages opt out
// of Apple's display-headroom tone mapping and clip bright windows to white.
inline CGImageRef CreateDisplayImage(const std::vector<uint8_t> &pixels, size_t width, size_t height, bool hdr) {
  CFDataRef data = CFDataCreate(nullptr, pixels.data(), pixels.size());
  CGDataProviderRef provider = CGDataProviderCreateWithCFData(data);
  CGColorSpaceRef space = CGColorSpaceCreateWithName(hdr ? kCGColorSpaceExtendedLinearSRGB : kCGColorSpaceSRGB);
  CGImageRef image;
  if (hdr) {
    float headroom = 1;
    for (size_t i = 0; i < pixels.size(); i += 16) {
      float rgb[3];
      std::memcpy(rgb, pixels.data() + i, sizeof(rgb));
      for (float value : rgb)
        headroom = std::max(headroom, value);
    }
    image = CGImageCreateWithContentHeadroom(headroom, width, height, 32, 128, width * 16, space,
                                             kCGBitmapFloatComponents | kCGBitmapByteOrder32Little | kCGImageAlphaLast,
                                             provider, nullptr, false, kCGRenderingIntentDefault);
  } else {
    image = CGImageCreate(width, height, 8, 32, width * 4, space, kCGBitmapByteOrderDefault | kCGImageAlphaLast,
                          provider, nullptr, false, kCGRenderingIntentDefault);
  }
  CGColorSpaceRelease(space);
  CGDataProviderRelease(provider);
  CFRelease(data);
  return image;
}
