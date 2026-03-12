#include <jni.h>
#include <string>
#include "ArcheryCounterModule.hpp"

// JNI function name must match: Java_<package>_<class>_<method>
// Package: com.helloworld  Class: ArcheryCounterModule  Method: nativeProcessImage
extern "C" JNIEXPORT jstring JNICALL
Java_com_helloworld_ArcheryCounterModule_nativeProcessImage(
    JNIEnv* env, jobject /* thiz */, jstring imagePath)
{
    const char* path = env->GetStringUTFChars(imagePath, nullptr);
    ArcheryResult result = processImageFile(std::string(path));
    env->ReleaseStringUTFChars(imagePath, path);

    if (!result.success) {
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, result.error.c_str());
        return nullptr;
    }

    // Return a JSON array string — parsed in Java to avoid complex JNI object construction.
    std::string json = "[";
    for (int i = 0; i < 10; i++) {
        const EllipseData& e = result.rings[i];
        if (i > 0) json += ",";
        json += "{\"centerX\":"  + std::to_string(e.centerX)
              + ",\"centerY\":" + std::to_string(e.centerY)
              + ",\"width\":"   + std::to_string(e.width)
              + ",\"height\":"  + std::to_string(e.height)
              + ",\"angle\":"   + std::to_string(e.angle) + "}";
    }
    json += "]";

    return env->NewStringUTF(json.c_str());
}
