#include "UVCCameraPlugin.h"
#include <cstdio>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

void* getCamera(int index)
{
    auto cap = new cv::VideoCapture(index);

    return static_cast<void*>(cap);
}

double getCameraProperty(void* camera, int propertyID) 
{
    auto cap = static_cast<cv::VideoCapture*>(camera);
    return cap->get(propertyID);
}

double setCameraProperty(void* camera, int propertyID, double value) 
{
    auto cap = static_cast<cv::VideoCapture*>(camera);
    return cap->set(propertyID, value);
}

void releaseCamera(void* camera)
{
    auto cap = static_cast<cv::VideoCapture*>(camera);
    delete cap;
}

double center_x = 0.0;
double center_y = 0.0;
double radius = 0.0;

void getCameraTexture(
    void* camera,
    unsigned char* data,
    bool executeHT21,
    bool executeMedianBlur,
    int imgMode,            // 0: src, 1: red, 2: green, 3: blue, 4: normalgray, 5: customgray 6: customgrayWithCircleOverlay
    double dp,
    double minDist,
    double param1,
    double param2,
    int minRadius,
    int maxRadius,
    bool bayer)
{
    auto cap = static_cast<cv::VideoCapture*>(camera);

    cv::Mat img;
    if (!cap->read(img)) {
        return;
    }

    Mat src;
   
    if (bayer) {
        // Convert 12-bit (encoded as 16-bit) pixels to 8-bit
        assert(img.type() == CV_16UC1);
        convertScaleAbs(img, src, 0.5);// 0.249023);
        assert(src.type() == CV_8UC1);

        // Copy the first G from the RGIB Bayer pixels into the second to get RGGB
        for (int i = 0; i < src.rows; i += 2) {
            const uchar *inRow = src.ptr<uchar>(i);
            uchar *outRow = src.ptr<uchar>(i + 1);

            for (int j = 0; j < src.cols; j += 2) {
                outRow[j] = inRow[j + 1];
            }
        }

        // De-bayer
        cv::cvtColor(src, src, COLOR_BayerRG2BGR);
    }
    else {
        src = img;
    }

    Mat bgr[3];
    split(src, bgr);

    Mat r = Mat(src.rows, src.cols, CV_8U, bgr[2].data);
    Mat g = Mat(src.rows, src.cols, CV_8U, bgr[1].data);
    Mat b = Mat(src.rows, src.cols, CV_8U, bgr[0].data);


    Mat gray;
    Mat normalGray;

	if (imgMode == 4)
    {
        cv::cvtColor(src, normalGray, COLOR_BGR2GRAY);
    }
    gray = g;

    if (executeHT21)
    {
        if (executeMedianBlur)
        {
            medianBlur(gray, gray, 5);
        }

        vector<Vec3f> circles;
        HoughCircles(
            gray,             // inputArray
            circles,          // outputArray
            HOUGH_GRADIENT,   // method
            dp,               // dp
            minDist,          // minDist
            param1,           // param1
            param2,           // param2
            minRadius,        // minRadius
            maxRadius         // maxRadius
        );

        if(circles.size() > 0)
        {
            Vec3i c = circles[0];
            center_x = (double)c[0];
            center_y = (double)c[1];
            radius = (double)c[2];
        }
        else
        {
            center_x = 0.0;
            center_y = 0.0;
            radius = 0.0;
        }

        if (imgMode == 6)
        {
            for (size_t i = 0; i < circles.size(); i++)
            {
                Vec3i c = circles[i];
                Point center = Point(c[0], c[1]);
                int radius = c[2];

				circle(gray, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
				circle(gray, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
            }
        }
    }

    // 0: src, 1: red, 2: green, 3: blue, 4: normalgray, 5: customgray
    cv::Mat rgba;
    if (imgMode == 0)
    {
        cv::cvtColor(src, rgba, cv::COLOR_BGR2RGBA);
    }
    else if (imgMode == 1)
    {
        cv::cvtColor(r, rgba, cv::COLOR_GRAY2RGBA);
    }
    else if (imgMode == 2)
    {
        cv::cvtColor(g, rgba, cv::COLOR_GRAY2RGBA);
    }
    else if (imgMode == 3)
    {
        cv::cvtColor(b, rgba, cv::COLOR_GRAY2RGBA);
    }
    else if (imgMode == 4)
    {
        cv::cvtColor(normalGray, rgba, cv::COLOR_GRAY2RGBA);
    }
    else if (imgMode == 5 || imgMode == 6)
    {
        cv::cvtColor(gray, rgba, cv::COLOR_GRAY2RGBA);
    }
    std::memcpy(data, rgba.data, rgba.total() * rgba.elemSize());
}

double getCircleCenter_x()
{
    return center_x;
}

double getCircleCenter_y()
{
    return center_y;
}

double getCircleRadius()
{
    return radius;
}
