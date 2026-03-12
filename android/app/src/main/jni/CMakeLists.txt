cmake_minimum_required(VERSION 3.13)
project(archerycounter_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# OpenCV Android SDK
# Download from https://opencv.org/releases/ (Android pack, e.g. 4.9.0)
# and unzip alongside the repo. Override this path via gradle argument:
#   cmake { arguments "-DOPENCV_ANDROID_SDK=/your/path" }
# ---------------------------------------------------------------------------
set(OPENCV_ANDROID_SDK "/Users/francoisancel/Documents/OpenCV-android-sdk"
    CACHE PATH "Root of the OpenCV Android SDK")

set(OPENCV_INCLUDE_DIR "${OPENCV_ANDROID_SDK}/sdk/native/jni/include")
set(OPENCV_LIB_DIR     "${OPENCV_ANDROID_SDK}/sdk/native/libs/${ANDROID_ABI}")

add_library(opencv_java4 SHARED IMPORTED)
set_target_properties(opencv_java4 PROPERTIES
    IMPORTED_LOCATION "${OPENCV_LIB_DIR}/libopencv_java4.so")

# ---------------------------------------------------------------------------
# Repo root: 5 levels up from android/app/src/main/jni/
# ---------------------------------------------------------------------------
set(CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../")

add_library(
    archerycounter_jni
    SHARED
    ArcheryCounterJNI.cpp
    ${CPP_ROOT}/cpp/ArcheryCounterModule.cpp
    ${CPP_ROOT}/native/ellispes_approach.cpp
    ${CPP_ROOT}/native/colour_filters.cpp
    ${CPP_ROOT}/native/pretreatment.cpp
)

target_include_directories(
    archerycounter_jni
    PRIVATE
    ${OPENCV_INCLUDE_DIR}
    ${CPP_ROOT}/native  # resolves "ellispes_approach.hpp", "math_utils.hpp", etc.
    ${CPP_ROOT}/cpp     # resolves "ArcheryCounterModule.hpp"
)

target_link_libraries(
    archerycounter_jni
    opencv_java4
    android
    log
    # With ANDROID_STL=c++_shared (set in build.gradle), libc++ provides
    # std::async / std::thread support without extra flags.
)
