package com.example.administrator.test;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CvCameraViewListener2 {

    static {
        System.loadLibrary(TFManager.LIBRARYLOCATION);
    }

    private static final String TAG = "Opencv_test" ;

    private CameraBridgeViewBase mOpenCvCameraView;     //将传递给TFManager对象

    private TFManager mTFManager;

    @Override
    public void onCreate(Bundle savedInstanceState) {
      //  Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            //申请权限
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
        }

        initViews();
        initWithTensorflow();    //由于要传递OpenCvCameraView对象，所以initWithTensorflow必须在initViews后面执行
    }



    private void initViews(){
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);                                    //设置Listener
        mOpenCvCameraView.setMaxFrameSize(640,640);
    }

    private void initWithTensorflow(){
        mTFManager = TFManager.getInstance(this);
        mTFManager.setCameraBridgeViewBase(mOpenCvCameraView);
        mTFManager.setOpenCVCallback(new BaseLoaderCallback(this) {                 //OpenCV回调函数对象的初始化
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                    {
                        //   Log.i(TAG, "OpenCV loaded successfully");
                        mOpenCvCameraView.enableView();
                    } break;
                    default:
                    {
                        super.onManagerConnected(status);
                    } break;
                }
            }
        });
        mTFManager.init();
    }


    @Override
    public void onResume()
    {
        super.onResume();
        mTFManager.resume();
    }

    @Override
    public void onPause()
    {
        super.onPause();
       mTFManager.pause();
    }

    public void onDestroy() {
        super.onDestroy();
        mTFManager.stop();
    }


    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }


    //CvCameraViewListener2接口函数
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat img_rgb = inputFrame.rgba();
        Mat img_t = new Mat();
        Mat img_gray = new Mat();
        Mat img_contours;

        Core.transpose(img_rgb,img_t);//转置函数，可以水平的图像变为垂直
        Imgproc.resize(img_t, img_rgb, img_rgb.size(), 0.0D, 0.0D, 0);
        Core.flip(img_rgb, img_rgb,1);  //flipCode>0将mRgbaF水平翻转（沿Y轴翻转）得到mRgba

        if(img_rgb != null) {
            Imgproc.cvtColor(img_rgb, img_gray, Imgproc.COLOR_RGB2GRAY);

            Imgproc.threshold(img_gray, img_gray, 140, 255, Imgproc.THRESH_BINARY_INV);

            //像素加强
            Mat ele1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            Mat ele2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6));
            Imgproc.erode(img_gray, img_gray, ele1);
            Imgproc.dilate(img_gray, img_gray, ele2);

            //找到外界矩形
            img_contours = img_gray.clone();
            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(img_contours, contours, new Mat(),
                    Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

            for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
                double contourArea = Imgproc.contourArea(contours.get(contourIdx));
                Rect rect = Imgproc.boundingRect(contours.get(contourIdx));
                if (contourArea < 1500 || contourArea > 20000)
                    continue;

                Mat roi = new Mat(img_gray, rect);
                Imgproc.resize(roi, roi, new Size(28, 28));

               mTFManager.detector(img_rgb);
            }
            img_contours.release();
        }


        img_gray.release();
        img_t.release();
        img_t.release();
        return  img_rgb;
    }
}
