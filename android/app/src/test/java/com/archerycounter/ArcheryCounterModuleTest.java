package com.archerycounter;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import android.content.ContentResolver;
import android.content.Context;
import android.net.Uri;

import com.facebook.react.bridge.ReactApplicationContext;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.io.ByteArrayInputStream;

@RunWith(RobolectricTestRunner.class)
@Config(manifest = Config.NONE)
public class ArcheryCounterModuleTest {

    private ArcheryCounterModule module;

    @Before
    public void setUp() {
        ReactApplicationContext ctx = mock(ReactApplicationContext.class);
        module = new ArcheryCounterModule(ctx);
    }

    @Test
    public void getName_returnsArcheryCounter() {
        assertEquals("ArcheryCounter", module.getName());
    }

    @Test
    public void resolveToFilePath_fileUri_stripsScheme() throws Exception {
        String result = module.resolveToFilePath("file:///sdcard/photo.jpg");
        assertEquals("/sdcard/photo.jpg", result);
    }

    @Test
    public void resolveToFilePath_plainPath_unchanged() throws Exception {
        String result = module.resolveToFilePath("/data/local/tmp/img.jpg");
        assertEquals("/data/local/tmp/img.jpg", result);
    }

    @Test
    public void resolveToFilePath_contentUri_copiesAndReturnsTmpPath() throws Exception {
        // Set up a mock Context and ContentResolver that returns a small stream
        ReactApplicationContext ctx = mock(ReactApplicationContext.class);
        ContentResolver resolver = mock(ContentResolver.class);
        byte[] fakeImage = new byte[]{(byte) 0xFF, (byte) 0xD8}; // JPEG magic bytes
        when(ctx.getContentResolver()).thenReturn(resolver);
        when(resolver.openInputStream(Uri.parse("content://media/photo.jpg")))
                .thenReturn(new ByteArrayInputStream(fakeImage));

        java.io.File cacheDir = java.io.File.createTempFile("test_cache", "").getParentFile();
        when(ctx.getCacheDir()).thenReturn(cacheDir);

        ArcheryCounterModule m = new ArcheryCounterModule(ctx);
        String path = m.resolveToFilePath("content://media/photo.jpg");

        // Should be a real file path (not a content:// URI)
        assertEquals(false, path.startsWith("content://"));
        assertEquals(true, new java.io.File(path).exists());
    }
}
