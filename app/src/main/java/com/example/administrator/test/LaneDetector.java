package com.example.administrator.test;

import android.app.Application;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Created by 11111 on 2018/5/3.
 */

public class LaneDetector extends Application{

    private Mat img_hsv;
    private Mat ele1,ele2;
    private Mat imgThresholded;
    private Mat img_line;
    private final String TAG="opencv";
    //左边线的斜率  右边线的斜率
    private double slope_left = 0,slope_right = 0;
    //是否识别到左边的线   是否识别到右边的线
    private double laneResult_left = -1,laneResult_right = -1;
    private int result = 0;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public LaneDetector(){

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        img_hsv = new Mat();
        imgThresholded = new Mat();
        img_line = new Mat();

        ele1= Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6,6));
        ele2= Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(4,4));
    }

    public int Detector(Mat img_rgb) {
        int r = img_rgb.cols()/2;
        Rect rect_left = new Rect(0,0,r,img_rgb.rows());
        Rect rect_right = new Rect(r,0,r,img_rgb.rows());
        Mat img_left = new Mat(img_rgb, rect_left);
        Mat img_right = new Mat(img_rgb, rect_right);


        Log.d("opencv   " , img_rgb.rows() +"   "+img_rgb.cols());


        Imgproc.cvtColor(img_left, img_hsv, Imgproc.COLOR_RGB2HSV);


        // 橙色11-25   黄色 26-34
        Core.inRange(img_hsv, new Scalar(11, 43, 46), new Scalar(34, 255, 255),
                imgThresholded);

        Imgproc.erode(imgThresholded, imgThresholded, ele1);
        Imgproc.dilate(imgThresholded, imgThresholded, ele2);



        //统计霍夫变换  距离分辨率 角度分辨率  直线点数阈值 长度阈值  线段最近两点的阈值
        Imgproc.HoughLinesP(imgThresholded, img_line, 4, Math.PI / 180,
                30, 60, 180);

        if(img_line.cols()>0 && img_line.rows()>0) {

            for (int i = 0; i < img_line.cols(); i++) {
                double[] line = img_line.get(0, i);
                if (line.length == 4) {
                    Imgproc.line(img_rgb, new Point(line[0], line[1]), new Point(line[2], line[3]),
                            new Scalar(0, 0, 255), 4);
                }
            }
        }

        getSlope(img_line, false);


        Imgproc.cvtColor(img_right, img_hsv, Imgproc.COLOR_RGB2HSV);
        // 绿色35-77   青色 78-99  蓝色 100-124
        Core.inRange(img_hsv, new Scalar(50, 43, 46), new Scalar(100, 255, 255),
                imgThresholded);

        Imgproc.erode(imgThresholded, imgThresholded, ele1);
        Imgproc.dilate(imgThresholded, imgThresholded, ele2);

        //统计霍夫变换  距离分辨率 角度分辨率  直线点数阈值 长度阈值  线段最近两点的阈值
        Imgproc.HoughLinesP(imgThresholded, img_line, 4, Math.PI / 180,
                30, 60, 180);

        if(img_line.cols()>0 && img_line.rows()>0) {

            for (int i = 0; i < img_line.cols(); i++) {
                double[] line = img_line.get(0, i);
                if (line.length == 4) {
                    Imgproc.line(img_rgb, new Point(line[0]+r, line[1]), new Point(line[2]+r, line[3]),
                            new Scalar(255, 0, 0), 4);
                }
            }
        }
        getSlope(img_line, true);


        img_left.release();
        img_right.release();

        /*
        逻辑判断   左转 右转  返回result
        1 2  左右转、
        3 直行
        4 停止
         */

        return result;
    }

    public void getSlope(Mat img_line, boolean flag){
        double slope = 0;
        double laneResult = -1;
        Mat dst = new Mat();

        if (img_line.cols() > 0 && img_line.rows()>0) {
            double det_slope = 0.4;
            int cnt = 0;
            for (int i = 0; i < img_line.cols(); i++) {
                double[] line = img_line.get(0, i);
                laneResult = 1;
                if (line[0] != line[2] && line[1] != line[3]) {
                    double det = (line[3] - line[1]) / (line[2] - line[0]);
                    if (Math.abs(det) > det_slope) {
                        slope += det;
                        cnt++;
                    }
                }
            }

            if (cnt != 0) {
                slope /= cnt;
            }
        }

        if(!flag)
        {
            slope_left = slope;
            laneResult_left = laneResult;
        }
        else {
            slope_right = slope;
            laneResult_right = laneResult;
        }

            /*if (cnt != 0) {
                slope /= cnt;
                List<MatOfPoint> contours = new ArrayList<>();
                Imgproc.findContours(imgThresholded, contours, dst, Imgproc.RETR_LIST,
                        Imgproc.CHAIN_APPROX_SIMPLE);
                if (contours.size() > 0) {
                    int mID = 0;

                    double mA = Imgproc.contourArea(contours.get(0));
                    for (int i = 1; i < contours.size(); i++) {
                        double a = Imgproc.contourArea(contours.get(i));
                        if (mA < a) {
                            mA = a;
                            mID = i;
                        }
                    }

                    //m00  面积   m10重心
                    Moments M = Imgproc.moments(contours.get(mID));
                    int cx = (int) (M.get_m10() / M.get_m00());
                    int cy = (int) (M.get_m01() / M.get_m00());

                    double b = cy - slope * cx;
                    left_lane = (480 - b) / slope;
                }
            }
        }*/

        dst.release();


        /*if(avgDetW <0)
            return 1;
        //左转 在道路的右侧
        if (avgDetY > 0 )
            return -1;
//右转 右边的线没识别成功
        if (targetLX1 !=-1 && targetRX1 == -1)
            return 1;
        // 两条线都没有识别到或者都识别到了
        if (targetLX1 !=-1 && targetRX1 ==-1 || targetLX1 != -1 && targetRX1!=-1){
            return 0;
        }
        //左边的线没识别成功，左转
        if(targetLX1 ==-1 && targetRX1!=-1)
        {
            return -1;
        }*/

    }

}
