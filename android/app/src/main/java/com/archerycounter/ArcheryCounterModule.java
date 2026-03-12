package com.archerycounter;

import android.content.Context;
import android.net.Uri;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class ArcheryCounterModule extends ReactContextBaseJavaModule {

    static {
        System.loadLibrary("archerycounter_jni");
    }

    public ArcheryCounterModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @Override
    public String getName() {
        // Must match RCT_EXPORT_MODULE(ArcheryCounter) on iOS and
        // NativeModules.ArcheryCounter on the JS side.
        return "ArcheryCounter";
    }

    /** JNI bridge — implemented in ArcheryCounterJNI.cpp */
    private native String nativeProcessImage(String filePath);

    @ReactMethod
    public void processImage(String imageUri, Promise promise) {
        // Run on a new thread — never block the React Native bridge thread.
        new Thread(() -> {
            try {
                String filePath = resolveToFilePath(imageUri);
                String json = nativeProcessImage(filePath);

                JSONArray arr = new JSONArray(json);
                WritableArray rings = Arguments.createArray();
                for (int i = 0; i < arr.length(); i++) {
                    JSONObject obj = arr.getJSONObject(i);
                    WritableMap ring = Arguments.createMap();
                    ring.putDouble("centerX", obj.getDouble("centerX"));
                    ring.putDouble("centerY", obj.getDouble("centerY"));
                    ring.putDouble("width",   obj.getDouble("width"));
                    ring.putDouble("height",  obj.getDouble("height"));
                    ring.putDouble("angle",   obj.getDouble("angle"));
                    rings.pushMap(ring);
                }
                promise.resolve(rings);

            } catch (Exception e) {
                promise.reject("PROCESSING_ERROR", e.getMessage());
            }
        }).start();
    }

    /**
     * Converts any URI scheme to a plain filesystem path that cv::imread can use.
     * - file:///path  →  /path
     * - content://    →  copy to cache, return cache path
     *   (required for Android 10+ scoped storage and some gallery providers)
     */
    String resolveToFilePath(String imageUri) throws Exception {
        if (imageUri.startsWith("file://")) {
            return imageUri.substring(7);
        }

        if (imageUri.startsWith("content://")) {
            Context ctx = getReactApplicationContext();
            Uri uri = Uri.parse(imageUri);
            InputStream is = ctx.getContentResolver().openInputStream(uri);
            if (is == null) throw new RuntimeException("Cannot open content URI: " + imageUri);

            File tmpFile = File.createTempFile("archery_img_", ".jpg", ctx.getCacheDir());
            try (FileOutputStream fos = new FileOutputStream(tmpFile)) {
                byte[] buf = new byte[8192];
                int n;
                while ((n = is.read(buf)) != -1) fos.write(buf, 0, n);
            }
            is.close();
            return tmpFile.getAbsolutePath();
        }

        // Assume it's already a plain path
        return imageUri;
    }
}
