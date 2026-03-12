require "json"

Pod::Spec.new do |s|
  s.name         = "ArcheryCounterNative"
  s.version      = "0.0.1"
  s.summary      = "OpenCV-based archery target scoring native module for React Native"
  s.homepage     = "https://github.com/Surrog/ArcheryCounter"
  s.license      = { :type => "MIT" }
  s.author       = { "ArcheryCounter" => "" }
  s.platform     = :ios, "13.0"
  s.source       = { :path => "." }

  s.source_files = [
    # iOS bridge
    "ios/ArcheryCounterBridge/*.{h,m,mm}",
    # Platform-independent C++ entry point
    "cpp/*.{hpp,cpp}",
    # Algorithm files (project root) — exclude main.cpp and unused approaches
    "ellispes_approach.{hpp,cpp}",
    "colour_filters.{hpp,cpp}",
    "pretreatment.{hpp,cpp}",
    "math_utils.hpp",
    "NN_approach.hpp",
  ]

  s.pod_target_xcconfig = {
    # The repo root must be on the search path so that
    # #include "ellispes_approach.hpp" resolves from cpp/ArcheryCounterModule.cpp
    # and from ios/ArcheryCounterBridge/RCTArcheryCounter.mm.
    "HEADER_SEARCH_PATHS"          => '"$(PODS_TARGET_SRCROOT)" "$(PODS_TARGET_SRCROOT)/cpp"',
    "CLANG_CXX_LANGUAGE_STANDARD"  => "c++17",
    "CLANG_CXX_LIBRARY"            => "libc++",
    # std::async on iOS uses GCD/libc++ — no extra linker flags needed.
  }

  s.dependency "React-Core"
  s.dependency "OpenCV2", "~> 4.8"
end
