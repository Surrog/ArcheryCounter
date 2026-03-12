// Objective-C++ — the .mm extension lets Xcode compile C++ and Obj-C together.
// The header search path for this file must include the repo root so that
// "ArcheryCounterModule.hpp" and "ellispes_approach.hpp" can be resolved.
// This is configured via ArcheryCounterNative.podspec at the repo root.

#import "RCTArcheryCounter.h"
#import "ArcheryCounterModule.hpp"

@implementation RCTArcheryCounter

// Exposes the module as NativeModules.ArcheryCounter on the JS side.
RCT_EXPORT_MODULE(ArcheryCounter)

RCT_EXPORT_METHOD(processImage:(NSString *)imageUri
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
    // Run on a background thread — never block the React Native bridge thread
    // with a potentially long OpenCV computation.
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{

        // react-native-image-picker returns "file:///absolute/path" on iOS.
        // cv::imread needs a plain filesystem path without the scheme prefix.
        NSString *path = imageUri;
        if ([path hasPrefix:@"file://"]) {
            path = [path substringFromIndex:7];
        }

        ArcheryResult result = processImageFile(std::string([path UTF8String]));

        if (!result.success) {
            reject(
                @"PROCESSING_ERROR",
                [NSString stringWithUTF8String:result.error.c_str()],
                nil
            );
            return;
        }

        NSMutableArray<NSDictionary *> *rings = [NSMutableArray arrayWithCapacity:10];
        for (int i = 0; i < 10; i++) {
            const EllipseData &e = result.rings[i];
            [rings addObject:@{
                @"centerX": @(e.centerX),
                @"centerY": @(e.centerY),
                @"width":   @(e.width),
                @"height":  @(e.height),
                @"angle":   @(e.angle),
            }];
        }

        resolve(rings);
    });
}

@end
