{
  "targets": [{
    "target_name": "Iir1",
    "cflags!" : [
      "-fno-exceptions"
    ],
    "cflags_cc!": [
      "-fno-rtti",
      "-fno-exceptions"
    ],
    "conditions": [
      ['OS=="mac"', {
        "xcode_settings": {
          "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        }
      }]
    ],
    "sources": [
      "src/main.cpp",
      "src/filters.cpp",
      "src/iir1/iir/Biquad.cpp",
      "src/iir1/iir/Butterworth.cpp",
      "src/iir1/iir/Cascade.cpp",
      "src/iir1/iir/ChebyshevI.cpp",
      "src/iir1/iir/ChebyshevII.cpp",
      "src/iir1/iir/Custom.cpp",
      "src/iir1/iir/PoleFilter.cpp",
      "src/iir1/iir/RBJ.cpp",
      "src/iir1/iir/State.cpp",
    ],
    'include_dirs': [
      "<!@(node -p \"require('node-addon-api').include\")",
      "cppsrc/"
    ],
    'libraries': [],
    'dependencies': [
      "<!(node -p \"require('node-addon-api').gyp\")"
    ],
    "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ]
  }]
}
