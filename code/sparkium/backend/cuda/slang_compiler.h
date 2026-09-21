#pragma once
#include <slang-com-ptr.h>
#include <slang.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace sparkium::backend::cuda {
std::string Read(const std::filesystem::path &path);
void Write(const std::filesystem::path &path, const std::string &source);
void SlangCheck(SlangResult result, const std::string &message);
SlangSession *Session();

struct RequestOwner {
  SlangCompileRequest *request = spCreateCompileRequest(Session());

  RequestOwner() {
    if (!request)
      throw std::runtime_error("failed to create Slang compile request");
  }

  ~RequestOwner() {
    spDestroyCompileRequest(request);
  }
};

}  // namespace sparkium::backend::cuda
