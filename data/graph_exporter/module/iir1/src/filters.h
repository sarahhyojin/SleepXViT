#include <napi.h>
#include "iir1/Iir.h"

namespace Filters {
  class HighPass : public Napi::ObjectWrap<HighPass> {
    public:
      static Napi::Object Init(Napi::Env env, Napi::Object exports);
      HighPass(const Napi::CallbackInfo& info);
    private:
      static Napi::FunctionReference constructor;
      void Setup(const Napi::CallbackInfo& info);
      Napi::Value Filter(const Napi::CallbackInfo& info);
      struct Iir::Butterworth::HighPass<4> *highPass;
  };

  class LowPass : public Napi::ObjectWrap<LowPass> {
    public:
      static Napi::Object Init(Napi::Env env, Napi::Object exports);
      LowPass(const Napi::CallbackInfo& info);
    private:
      static Napi::FunctionReference constructor;
      void Setup(const Napi::CallbackInfo& info);
      Napi::Value Filter(const Napi::CallbackInfo& info);
      struct Iir::Butterworth::LowPass<4> *lowPass;
  };

  class BandStop : public Napi::ObjectWrap<BandStop> {
    public:
      static Napi::Object Init(Napi::Env env, Napi::Object exports);
      BandStop(const Napi::CallbackInfo& info);
    private:
      static Napi::FunctionReference constructor;
      void Setup(const Napi::CallbackInfo& info);
      Napi::Value Filter(const Napi::CallbackInfo& info);
      struct Iir::Butterworth::BandStop<4> *bandStop;
  };
}
