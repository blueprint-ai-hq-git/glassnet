package com.glassnet.math1;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;

/**
 * GlassNet Runner - a LOCAL app that loads a separately supplied model file.
 *
 * No INTERNET permission is declared in the manifest, so this app cannot open a network connection at all. It shows a page that is
 * packaged inside the app and reads a model file the user picks from their own storage. The model is never bundled.
 */
public class MainActivity extends Activity {
    private static final int PICK_MODEL = 1001;
    private WebView wv;
    private ValueCallback<Uri[]> pending;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        wv = new WebView(this);
        WebSettings s = wv.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(true);
        wv.setBackgroundColor(Color.parseColor("#0b0d10"));
        // the file picker for <input type=file> - without this, "Open model" would do nothing on Android
        wv.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> cb, FileChooserParams params) {
                pending = cb;
                Intent i = new Intent(Intent.ACTION_GET_CONTENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType("*/*");
                try {
                    startActivityForResult(Intent.createChooser(i, "Pick a GlassNet model"), PICK_MODEL);
                    return true;
                } catch (Exception e) {
                    pending = null;
                    return false;
                }
            }
        });
        wv.loadUrl("file:///android_asset/index.html");
        setContentView(wv);
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        if (req == PICK_MODEL) {
            Uri[] out = null;
            if (res == RESULT_OK && data != null && data.getData() != null) out = new Uri[]{ data.getData() };
            if (pending != null) { pending.onReceiveValue(out); pending = null; }
            return;
        }
        super.onActivityResult(req, res, data);
    }

    @Override
    public void onBackPressed() {
        if (wv != null && wv.canGoBack()) wv.goBack(); else super.onBackPressed();
    }
}
