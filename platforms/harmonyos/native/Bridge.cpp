#include <ace/xcomponent/native_interface_xcomponent.h>
#include <napi/native_api.h>

#include "Host.h"

using longmarch::harmony::Host;

namespace {
std::string String(napi_env env, napi_value value) {
  size_t size = 0;
  if (napi_get_value_string_utf8(env, value, nullptr, 0, &size) != napi_ok)
    throw std::runtime_error("Expected string argument");
  std::string result(size + 1, '\0');
  if (napi_get_value_string_utf8(env, value, result.data(), result.size(), &size) != napi_ok)
    throw std::runtime_error("Cannot read string argument");
  result.resize(size);
  return result;
}

napi_value Undefined(napi_env env) {
  napi_value value;
  napi_get_undefined(env, &value);
  return value;
}

napi_value Initialize(napi_env env, napi_callback_info info) {
  try {
    size_t count = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &count, args, nullptr, nullptr);
    if (count != 2)
      throw std::runtime_error("initialize requires resource manager and files directory");
    auto directory = String(env, args[1]);
    auto *manager = OH_ResourceManager_InitNativeResourceManager(env, args[0]);
    if (!manager)
      throw std::runtime_error("Cannot initialize native resource manager");
    Host::Get().Initialize(manager, directory);
  } catch (const std::exception &error) {
    napi_throw_error(env, nullptr, error.what());
  }
  return Undefined(env);
}

napi_value Command(napi_env env, napi_callback_info info) {
  try {
    size_t count = 1;
    napi_value arg;
    napi_get_cb_info(env, info, &count, &arg, nullptr, nullptr);
    if (count != 1)
      throw std::runtime_error("command requires JSON string");
    Host::Get().Command(String(env, arg));
  } catch (const std::exception &error) {
    napi_throw_error(env, nullptr, error.what());
  }
  return Undefined(env);
}

napi_value Status(napi_env env, napi_callback_info) {
  auto text = Host::Get().Status();
  napi_value result;
  napi_create_string_utf8(env, text.c_str(), text.size(), &result);
  return result;
}

void SurfaceChanged(OH_NativeXComponent *component, void *window) {
  uint64_t width = 0, height = 0;
  if (OH_NativeXComponent_GetXComponentSize(component, window, &width, &height) == OH_NATIVEXCOMPONENT_RESULT_SUCCESS)
    Host::Get().Attach(static_cast<OHNativeWindow *>(window), width, height);
}

void SurfaceDestroyed(OH_NativeXComponent *, void *) {
  Host::Get().Detach();
}

void Touch(OH_NativeXComponent *, void *) {
}  // ArkUI sends normalized multi-touch events.

OH_NativeXComponent_Callback callbacks{SurfaceChanged, SurfaceChanged, SurfaceDestroyed, Touch};

napi_value Register(napi_env env, napi_value exports) {
  napi_property_descriptor properties[] = {
      {"initialize", nullptr, Initialize, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"command", nullptr, Command, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"status", nullptr, Status, nullptr, nullptr, nullptr, napi_default, nullptr}};
  napi_define_properties(env, exports, 3, properties);
  napi_value value;
  bool has = false;
  napi_has_named_property(env, exports, OH_NATIVE_XCOMPONENT_OBJ, &has);
  if (has && napi_get_named_property(env, exports, OH_NATIVE_XCOMPONENT_OBJ, &value) == napi_ok) {
    OH_NativeXComponent *component = nullptr;
    if (napi_unwrap(env, value, reinterpret_cast<void **>(&component)) == napi_ok && component)
      OH_NativeXComponent_RegisterCallback(component, &callbacks);
  }
  return exports;
}

napi_module module{1, 0, nullptr, Register, "longmarch", nullptr, {0}};
}  // namespace

extern "C" __attribute__((constructor)) void RegisterLongMarch() {
  napi_module_register(&module);
}
