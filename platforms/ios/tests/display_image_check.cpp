#include <cmath>
#include <iostream>
#include <stdexcept>

#include "DisplayImage.h"

int main() {
  try {
    // A bright window ramp must remain distinguishable on a lower-headroom
    // display, while the source floating-point radiance remains untouched.
    const float rgba[] = {.18f, .18f, .18f, 1, 2, 2, 2, 1, 8, 8, 8, 1, 20, 20, 20, 1};
    std::vector<uint8_t> pixels(sizeof(rgba));
    std::memcpy(pixels.data(), rgba, sizeof(rgba));
    CGImageRef image = CreateDisplayImage(pixels, 4, 1, true);
    if (!image || CGImageGetContentHeadroom(image) != 20 || !CGImageShouldToneMap(image))
      throw std::runtime_error("HDR image is not tagged for system tone mapping");
    auto source = CGDataProviderCopyData(CGImageGetDataProvider(image));
    if (CFDataGetLength(source) != sizeof(rgba) || std::memcmp(CFDataGetBytePtr(source), rgba, sizeof(rgba)))
      throw std::runtime_error("HDR radiance changed during image packaging");
    CFRelease(source);
    auto space = CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
    for (float headroom : {1.f, 2.f, 4.f}) {
      float output[16]{};
      auto context =
          CGBitmapContextCreate(output, 4, 1, 32, sizeof(output), space,
                                kCGBitmapFloatComponents | kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedLast);
      if (!context || !CGContextSetEDRTargetHeadroom(context, headroom))
        throw std::runtime_error("Cannot set target display headroom");
      CGContextDrawImage(context, CGRectMake(0, 0, 4, 1), image);
      CGContextRelease(context);
      for (int i = 1; i < 4; ++i)
        if (!std::isfinite(output[i * 4]) || output[i * 4] <= output[(i - 1) * 4] || output[i * 4] > headroom + .01f)
          throw std::runtime_error("Bright window ramp clipped or lost highlight ordering");
      std::cout << "PASS display headroom " << headroom << ": " << output[0] << ", " << output[4] << ", " << output[8]
                << ", " << output[12] << '\n';
    }
    CGColorSpaceRelease(space);
    CGImageRelease(image);
    auto sdr = CreateDisplayImage(std::vector<uint8_t>{128, 64, 32, 255}, 1, 1, false);
    if (!sdr || CGImageGetBitsPerComponent(sdr) != 8 || CGImageShouldToneMap(sdr))
      throw std::runtime_error("SDR image unexpectedly requests HDR tone mapping");
    CGImageRelease(sdr);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
