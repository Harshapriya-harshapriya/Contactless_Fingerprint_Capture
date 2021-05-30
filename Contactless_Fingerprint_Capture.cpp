#include <opencv2/opencv.hpp>
#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

int MAX_KERNEL_LENGTH = 31;


const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
const String window_filter_name = "contour_image";
const String window_blur_name = "bluredimage";
const String window_rectangle_drawn = "recatnlw";

int low_H = 0, low_S = 25, low_V = 50;
//int high_H = max_value_H, high_S = max_value, high_V = max_value;
int high_H = 25, high_S = 125, high_V = 225;

static void on_low_H_thresh_trackbar(int, void*)
{
    low_H = min(high_H - 1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
    high_H = max(high_H, low_H + 1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
    low_S = min(high_S - 1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
    high_S = max(high_S, low_S + 1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
    low_V = min(high_V - 1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
    high_V = max(high_V, low_V + 1);
    setTrackbarPos("High V", window_detection_name, high_V);
}


int main(int argc, char** argv)
{
    //#include "opencv2/imgproc.hpp"
    //#include "opencv2/highgui.hpp"
    //#include "opencv2/videoio.hpp"
    //#include <iostream>
        //using namespace cv;
    double beta, alpha = 0.5;

    VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
    namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    namedWindow(window_filter_name);
    //namedWindow(window_blur_name);
   // namedWindow(window_rectangle_drawn);
    // Trackbars to set thresholds for HSV values
    createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);

    Mat frame, blur_frame, frame_HSV, frame_threshold, dest, threshold_image;
    while (true) {
        cap >> frame;
        if (frame.empty())
            return -1;
        imshow(window_capture_name, frame);
        // blur(frame, blur_frame, Size(3, 3));
        // imshow(window_blur_name, blur_frame);
         // Convert from BGR to HSV colorspace

        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);


        // Detect the object based on HSV Range Values
       // inRange(frame_HSV, Scalar(0, 10, 60), Scalar(20, 150, 255), frame_threshold);
        inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
        imshow(window_detection_name, frame_threshold);
        for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
        {
            //bilateralFilter(frame_threshold, threshold_image, i, i * 2, i / 2);
            medianBlur(frame_threshold, threshold_image, i);

        }


        // imshow(window_filter_name, threshold_image);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(frame_threshold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, cv::Point(0, 0)); //finding contours in the binary image

        //drawing rectangle to the contours

        vector<RotatedRect> minRect(contours.size());

        for (size_t i = 0; i < contours.size(); i++)
        {
            minRect[i] = minAreaRect(contours[i]);
            minRect[i].size.height = minRect[i].size.width * 1.6;

        }
        Mat drawing = Mat::zeros(frame_threshold.size(), CV_8UC3);
        for (size_t i = 0; i < contours.size(); i++)
        {
            Scalar color = Scalar(255, 255, 255);
            // drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
            Point2f rect_points[4];

            minRect[i].points(rect_points);
            for (int j = 0; j < 4; j++)
            {
                line(drawing, rect_points[j], rect_points[(j + 1) % 4], color);
            }
        }
        //  imshow(window_filter_name, drawing);


        beta = (1.0 - alpha);
        addWeighted(frame, alpha, drawing, beta, 0.0, dest);
        imshow("image", dest);



        char key = (char)waitKey(30);
        if (key == 'q' || key == 27)
        {
            break;
        }


    }


    return 0;

}
