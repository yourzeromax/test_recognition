package com.example.administrator.test;

import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class test {

//    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
//        Mat img_rgb = inputFrame.rgba();
//        Mat img_t = new Mat();
//        Mat img_gray = new Mat();
//        Mat img_contours;
//
//        Core.transpose(img_rgb,img_t);//转置函数，可以水平的图像变为垂直
//        Imgproc.resize(img_t, img_rgb, img_rgb.size(), 0.0D, 0.0D, 0);
//        Core.flip(img_rgb, img_rgb,1);  //flipCode>0将mRgbaF水平翻转（沿Y轴翻转）得到mRgba
//
//        if(img_rgb != null) {
//            Imgproc.cvtColor(img_rgb, img_gray, Imgproc.COLOR_RGB2GRAY);
//
//            Imgproc.threshold(img_gray, img_gray, 140, 255, Imgproc.THRESH_BINARY_INV);
//
//            //像素加强
//            Mat ele1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
//            Mat ele2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6));
//            Imgproc.erode(img_gray, img_gray, ele1);
//            Imgproc.dilate(img_gray, img_gray, ele2);
//
//            //找到外界矩形
//            img_contours = img_gray.clone();
//            List<MatOfPoint> contours = new ArrayList<>();
//            Imgproc.findContours(img_contours, contours, new Mat(),
//                    Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
//
//            for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
//                double contourArea = Imgproc.contourArea(contours.get(contourIdx));
//                Rect rect = Imgproc.boundingRect(contours.get(contourIdx));
//                if (contourArea < 1500 || contourArea > 20000)
//                    continue;
//
//                Mat roi = new Mat(img_gray, rect);
//                Imgproc.resize(roi, roi, new Size(28, 28));
//
//                Bitmap bitmap2 = Bitmap.createBitmap(roi.width(), roi.height(), Bitmap.Config.RGB_565);
//                Utils.matToBitmap(roi, bitmap2);
//                int number = toNumber(bitmap2);
//                if (number >= 0) {
//                    //tl左上角顶点  br右下角定点                      //显示蓝色的检测结果
//                    double x = rect.tl().x;
//                    double y = rect.br().y;
//                    Point p = new Point(x, y);
//                    Imgproc.rectangle(img_rgb, rect.tl(), rect.br(), new Scalar(0, 0, 255));
//                    Imgproc.putText(img_rgb, Integer.toString(number), p, Core.FONT_HERSHEY_DUPLEX,
//                            6, new Scalar(0, 0, 255), 2);
//                }
//            }
//            img_contours.release();
//        }
//
//
//        img_gray.release();
//        img_t.release();
//        img_t.release();
//        return  img_rgb;
//    }
//
//    //28 X 28   1
//    //给一个 bitmap   识别成数字  以int形式返回
//    int toNumber(Bitmap bitmap_roi){
//        int width = bitmap_roi.getWidth();
//        int height = bitmap_roi.getHeight();
//        int[] pixels = new int[width * height];
//
//        Log.d("tag", width+"  "+height);
//
//        try {
//            bitmap_roi.getPixels(pixels, 0, width, 0, 0, width, height);
//            for (int i = 0; i < pixels.length; i++) {
//                inputs_data[i] = (float)pixels[i];
//            }
//        }catch (Exception e){
//            Log.d("tag", e.getMessage());
//        }
//
//        Log.d("Tag", "width: "+width+"   height:"+height);
//
//        Trace.beginSection("feed");
//        inferenceInterface.feed("conv2d_1_input_2:0", inputs_data, 1,28,28,1);
//        Trace.endSection();
//
//        Trace.beginSection("run");
//        inferenceInterface.run(new String[]{OUTPUT_NODE});
//        Trace.endSection();
//
//        Trace.beginSection("fetch");
//        inferenceInterface.fetch(OUTPUT_NODE, outputs_data);
//        Trace.endSection();
//
//        int logit = 0;
//        for(int i=1;i<10;i++)
//        {
//            if(outputs_data[i]>outputs_data[logit])
//                logit=i;
//        }
//
//        if(outputs_data[logit]>0)
//            return logit;
//        return -1;
//
//    }
}


