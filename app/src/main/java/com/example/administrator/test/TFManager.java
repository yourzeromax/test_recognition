package com.example.administrator.test;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
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
import java.util.Vector;

import static android.content.ContentValues.TAG;

public class TFManager {
    public static final String LIBRARYLOCATION ="tensorflow_inference";
    private static final String MODE_FILE = "file:///android_asset/ssd_mobilenet_v1_android_export.pb";             //模型的存储路径
    private  static final String LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

    private  static TFManager mTFManager;        //单例模式
    private OpenCVManager mOpenCVManager;
    private TensorFlowInferenceInterface tensorFlowInferenceInterface = null;

    private Context context;

    private final int inputSize = 300;
    private final String input_name = "image_tensor";
    private final String[] output_names = new String[] {"detection_boxes", "detection_scores", "detection_classes", "num_detections"};
    private boolean logStats = false;

    private static final int MAX_RESULTS = 100;
    private float[] outputLocations = new float[MAX_RESULTS*4];
    private float[] outputScores = new float[MAX_RESULTS];
    private float[] outputClasses = new float[MAX_RESULTS];
    private float[] outputNumDetections = new float[1];

    private Vector<String> labels = new Vector<String>();

    private TFManager(Context context){
        this.context = context;
        mOpenCVManager = new OpenCVManager(context);
        tensorFlowInferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODE_FILE);
    }

    public static TFManager getInstance(Context context) {
        if(mTFManager==null){
            synchronized (TFManager.class){
                if(mTFManager==null){
                    mTFManager = new TFManager(context);
                }
            }
        }
        return mTFManager;
    }


    public void init(){

        String line;
        try {
            InputStream labelsInput = null;
            String filename = LABELS_FILE.split("file:///android_asset/")[1];
            labelsInput = context.getAssets().open(filename);
            BufferedReader br = null;
            br = new BufferedReader(new InputStreamReader(labelsInput));
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
            br.close();
        }catch (IOException e){
            Log.d(TAG, e.toString());
        }
        mOpenCVManager.init();
    }

   public void start(){
       mOpenCVManager.init();
   }

   public void resume(){
        mOpenCVManager.resume();
   }

   public void pause(){
    mOpenCVManager.pause();
   }

   public  void stop(){
    mOpenCVManager.stop();
   }

   public void setOpenCVCallback(BaseLoaderCallback loaderCallback){
        mOpenCVManager.mLoaderCallback = loaderCallback;
   }

   public void setCameraBridgeViewBase(CameraBridgeViewBase mOpenCvCameraView){             //设置CameraBrigeViewBase的监听器
        mOpenCVManager.mOpenCvCameraView = mOpenCvCameraView;
   }

    void detector(Mat img_rgb){

        if(tensorFlowInferenceInterface == null){
            return ;}

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
