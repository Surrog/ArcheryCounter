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
    # Algorithm files — exclude main.cpp and unused approaches
    "native/ellispes_approach.{hpp,cpp}",
    "native/colour_filters.{hpp,cpp}",
    "native/pretreatment.{hpp,cpp}",
    "native/math_utils.hpp",
    "native/NN_approach.hpp",
  ]

  s.pod_target_xcconfig = {
    "HEADER_SEARCH_PATHS"          => '"$(PODS_TARGET_SRCROOT)/native" "$(PODS_TARGET_SRCROOT)/cpp"',
    "CLANG_CXX_LANGUAGE_STANDARD"  => "c++17",
    "CLANG_CXX_LIBRARY"            => "libc++",
    # std::async on iOS uses GCD/libc++ — no extra linker flags needed.
  }

  s.dependency "React-Core"
  # OpenCV-Dynamic-Framework ships as an xcframework with both device + simulator slices.
  # OpenCV2 (4.3) only has device slices and breaks simulator builds.
  s.dependency "OpenCV-Dynamic-Framework", "~> 4.10"
end
