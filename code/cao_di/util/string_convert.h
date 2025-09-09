#pragma once
#include "string"

namespace CD {
std::string WStringToString(const std::wstring &wstr);

std::wstring StringToWString(const std::string &str);

}  // namespace CD
