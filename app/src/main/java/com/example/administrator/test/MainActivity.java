package com.example.administrator.test;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Trace;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "Opencv_test" ;

    private CameraBridgeViewBase mOpenCvCameraView;

    static {
        System.loadLibrary(TFManager.LIBRARYLOCATION);
    }

    private TensorFlowInferenceInterface tensorFlowInferenceInterface = null;
    private  String labels_file = "file:///android_asset/coco_labels_list.txt";
    private static final String mode_file = "file:///android_asset/ssd_mobilenet_v1_android_export.pb";

    public final String input_name = "image_tensor";
    public final String[] output_names = new String[] {"detection_boxes", "detection_scores",
            "detection_classes", "num_detections"};
    public final int inputSize = 300;
    private boolean logStats = false;

    private static final int MAX_RESULTS = 100;
    float[] outputLocations = new float[MAX_RESULTS*4];
    float[] outputScores = new float[MAX_RESULTS];
    float[] outputClasses = new float[MAX_RESULTS];
    float[] outputNumDetections = new float[1];

    private Vector<String> labels = new Vector<String>();

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    /** Called when the activity is first created. */
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
        initWithTensorflow();
    }

    private void initViews(){
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(640,640);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private void initWithTensorflow(){
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }


        String line;
        try {
            InputStream labelsInput = null;
            String filename = labels_file.split("file:///android_asset/")[1];
            labelsInput = getAssets().open(filename);
            BufferedReader br = null;
            br = new BufferedReader(new InputStreamReader(labelsInput));
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
            br.close();
        }catch (IOException e){
            Log.d(TAG, e.toString());
        }

        tensorFlowInferenceInterface = new TensorFlowInferenceInterface(getAssets(), mode_file);
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

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

               Detector(img_rgb);
            }
            img_contours.release();
        }


        img_gray.release();
        img_t.release();
        img_t.release();
        return  img_rgb;
    }


    void Detector(Mat img_rgb){

        if(tensorFlowInferenceInterface == null)
            return ;

        Mat img = new Mat();
        Imgproc.resize(img_rgb, img, new Size(inputSize,inputSize));
        Bitmap bitmap = Bitmap.createBitmap(img.width(), img.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, bitmap);

        int[] intValues = new int[img.width() * img.height()];
        byte[] byteValues = new byte[img.width() * img.height() * 3];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(),
                0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            byteValues[i * 3] = (byte) ((val >> 16) & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((val >> 8) & 0xFF);
            byteValues[i * 3 + 2] = (byte) ((val & 0xFF));
        }

        Trace.beginSection("feed");
        tensorFlowInferenceInterface.feed(input_name, byteValues, 1, img.width(), img.height(), 3);
        Trace.endSection();

        Trace.beginSection("run");
        tensorFlowInferenceInterface.run(output_names, logStats);
        Trace.endSection();

        Trace.beginSection("fetch");
        tensorFlowInferenceInterface.fetch(output_names[0], outputLocations);
        tensorFlowInferenceInterface.fetch(output_names[1], outputScores);
        tensorFlowInferenceInterface.fetch(output_names[2], outputClasses);
        tensorFlowInferenceInterface.fetch(output_names[3], outputNumDetections);
        Trace.endSection();


        for (int i = 0; i < outputScores.length; ++i) {

            if(outputScores[i] > 0.6) {
                Point p1 = new Point(outputLocations[4 * i + 1] * inputSize,
                        outputLocations[4 * i] * inputSize);
                Point p2 = new Point(outputLocations[4 * i + 3] * inputSize,
                        outputLocations[4 * i + 2] * inputSize);

                final Rect detection = new Rect(p1, p2);

                Imgproc.rectangle(img, p1,p2,
                        new Scalar(0,0,255), 2);
                Imgproc.putText(img, labels.get((int) outputClasses[i]), p1, Core.FONT_HERSHEY_DUPLEX,
                        1, new Scalar(0, 255, 0), 1);
                Log.d(TAG, "            " + outputClasses[i]);
            }
        }

        Imgproc.resize(img, img_rgb, img_rgb.size());
        img.release();
    }

}
