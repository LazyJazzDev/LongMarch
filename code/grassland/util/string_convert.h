#pragma once
#include "string"

namespace grassland {
std::string WStringToString(const std::wstring &wstr);

std::wstring StringToWString(const std::string &str);

}  // namespace grassland
