#include "filters.h"
#include "iir1/Iir.h"

using namespace Filters;

/* Highpass Filter */
Napi::FunctionReference Filters::HighPass::constructor;

Napi::Object Filters::HighPass::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "HighPass", {
    InstanceMethod("setup", &HighPass::Setup),
    InstanceMethod("filter", &HighPass::Filter)
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("HighPass", func);
  return exports;
}

Filters::HighPass::HighPass(const Napi::CallbackInfo& info) : Napi::ObjectWrap<HighPass>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);
  this->highPass = new Iir::Butterworth::HighPass<4>;
}

void Filters::HighPass::Setup(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
    Napi::TypeError::New(env, "Number expected").ThrowAsJavaScriptException();
    return;
  } 

  Napi::Number samplingrate = info[0].As<Napi::Number>();
  Napi::Number cutoff_frequency = info[1].As<Napi::Number>();

  if (samplingrate.FloatValue() < cutoff_frequency.FloatValue()) {
    Napi::Error::New(env, "The cutoff frequency should be lower or equal to the sampling frequency").ThrowAsJavaScriptException();
    return;
  }

  this->highPass->setup(samplingrate.FloatValue(), cutoff_frequency.FloatValue());
}

Napi::Value Filters::HighPass::Filter(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Number expected").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Number x = info[0].As<Napi::Number>();

  double y = this->highPass->filter(x.DoubleValue());
  return Napi::Number::New(env, y);
}

/* Lowpass Filter */
Napi::FunctionReference Filters::LowPass::constructor;

Napi::Object Filters::LowPass::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "LowPass", {
    InstanceMethod("setup", &LowPass::Setup),
    InstanceMethod("filter", &LowPass::Filter)
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("LowPass", func);
  return exports;
}

Filters::LowPass::LowPass(const Napi::CallbackInfo& info) : Napi::ObjectWrap<LowPass>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);
  this->lowPass = new Iir::Butterworth::LowPass<4>;
}

void Filters::LowPass::Setup(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
    Napi::TypeError::New(env, "Number expected").ThrowAsJavaScriptException();
    return;
  } 

  Napi::Number samplingrate = info[0].As<Napi::Number>();
  Napi::Number cutoff_frequency = info[1].As<Napi::Number>();

  if (samplingrate.FloatValue() < cutoff_frequency.FloatValue()) {
    Napi::Error::New(env, "The cutoff frequency should be lower or equal to the sampling frequency").ThrowAsJavaScriptException();
    return;
  }

  this->lowPass->setup(samplingrate.FloatValue(), cutoff_frequency.FloatValue());
}

Napi::Value Filters::LowPass::Filter(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Number expected").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Number x = info[0].As<Napi::Number>();

  double y = this->lowPass->filter(x.DoubleValue());
  return Napi::Number::New(env, y);
}

/* Bandstop filter */
Napi::FunctionReference Filters::BandStop::constructor;

Napi::Object Filters::BandStop::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "BandStop", {
    InstanceMethod("setup", &BandStop::Setup),
    InstanceMethod("filter", &BandStop::Filter)
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("BandStop", func);
  return exports;
}

Filters::BandStop::BandStop(const Napi::CallbackInfo& info) : Napi::ObjectWrap<BandStop>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);
  this->bandStop = new Iir::Butterworth::BandStop<4>;
}

void Filters::BandStop::Setup(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (info.Length() < 3
      || !info[0].IsNumber()
      || !info[1].IsNumber()
      || !info[2].IsNumber()) {
    Napi::TypeError::New(env, "Number expected").ThrowAsJavaScriptException();
    return;
  } 

  Napi::Number samplingrate = info[0].As<Napi::Number>();
  Napi::Number center_frequency = info[1].As<Napi::Number>();
  Napi::Number width_frequency = info[2].As<Napi::Number>();

  if (samplingrate.FloatValue() < center_frequency.FloatValue()) {
    Napi::Error::New(env, "The cutoff frequency should be lower or equal to the sampling frequency").ThrowAsJavaScriptException();
    return;
  }

  this->bandStop->setup(
      samplingrate.FloatValue(),
      center_frequency.FloatValue(),
      width_frequency.FloatValue()
  );
}

Napi::Value Filters::BandStop::Filter(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Number expected").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Number x = info[0].As<Napi::Number>();

  double y = this->bandStop->filter(x.DoubleValue());
  return Napi::Number::New(env, y);
}
