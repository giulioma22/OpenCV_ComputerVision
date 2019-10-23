#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include "bg.h"

using namespace cv;

int main() {

    Mat frame, frame_gray;
    Mat bg, motion_mask, motion_mask_t;
    
    VideoCapture cap("Video.mp4");

    if (!cap.isOpened()) {
        return 0;
    }

    for (int i = 0; i < 1000; i++) {
        cap >> frame;
        if (i > 0) {
            //Color conv from RGB to gray
            cvtColor(frame, frame_gray, COLOR_RGB2GRAY);
            //Store 1st frame as background
            bg_train(frame_gray, &bg);  //Since it is a pointer, we need to put the address &
            bg_update(frame_gray, &bg);
            //BG subtraction
            absdiff(bg, frame_gray, motion_mask);

            //Thresholding
            threshold(motion_mask, motion_mask_t, 50, 255, THRESH_BINARY);

            //Display
            imshow("Original", frame);
            imshow("Background", bg);
            imshow("mption mask", motion_mask_t);
            waitKey(1);
        }

    }

    return 0;
}