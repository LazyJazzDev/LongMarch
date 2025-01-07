#include "grassland/util/string_convert.h"

#include "cstdint"

namespace grassland {
std::string WStringToString(const std::wstring &wstr) {
  // Convert the WString to String, encode as utf-8
  uint8_t c;
  std::string str;
  for (uint32_t wc : wstr) {
    if (wc < 0x80) {
      // 1 byte
      c = wc;
      str.push_back(c);
    } else if (wc < 0x800) {
      // 2 bytes
      c = 0xC0 | ((wc >> 6) & 0x1F);
      str.push_back(c);
      c = 0x80 | (wc & 0x3F);
      str.push_back(c);
    } else if (wc < 0x10000) {
      // 3 bytes
      c = 0xE0 | ((wc >> 12) & 0x0F);
      str.push_back(c);
      c = 0x80 | ((wc >> 6) & 0x3F);
      str.push_back(c);
      c = 0x80 | (wc & 0x3F);
      str.push_back(c);
    } else {
      // 4 bytes
      c = 0xF0 | ((wc >> 18) & 0x07);
      str.push_back(c);
      c = 0x80 | ((wc >> 12) & 0x3F);
      str.push_back(c);
      c = 0x80 | ((wc >> 6) & 0x3F);
      str.push_back(c);
      c = 0x80 | (wc & 0x3F);
      str.push_back(c);
    }
  }
  return str;
}

std::wstring StringToWString(const std::string &str) {
  // Convert String to WString, decode as utf-8
  std::wstring wstr;
  wchar_t wc = 0;
  uint32_t appendix_count = 0;
  for (uint8_t c : str) {
    if (c < 0x80) {
      wc = c;
      wstr.push_back(wc);
    } else if ((c & 0xE0) == 0xC0) {
      wc = c & 0x1F;
      appendix_count = 1;
    } else if ((c & 0xF0) == 0xE0) {
      wc = c & 0x0F;
      appendix_count = 2;
    } else if ((c & 0xF8) == 0xF0) {
      wc = c & 0x07;
      appendix_count = 3;
    } else {
      wc <<= 6;
      wc |= c & 0x3F;
      if (--appendix_count == 0) {
        wstr.push_back(wc);
      }
    }
  }
  return wstr;
}
}  // namespace grassland
