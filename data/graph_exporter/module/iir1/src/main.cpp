#include <napi.h>
#include "filters.h"

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  Filters::HighPass::Init(env, exports);
  Filters::LowPass::Init(env, exports);
  Filters::BandStop::Init(env, exports);
  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, InitAll);
