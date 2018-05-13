package com.example.administrator.test;

import android.content.Context;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import static android.content.ContentValues.TAG;

public class OpenCVManager {
   public BaseLoaderCallback mLoaderCallback;
    public CameraBridgeViewBase mOpenCvCameraView;

    private Context context;

    public void init(){
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, context, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

    }

    public void resume(){
        init();
    }

    public void pause(){
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void stop(){
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public OpenCVManager(Context context){
        this.context = context;
    }


}
