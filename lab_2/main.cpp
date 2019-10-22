#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main() {

    //How to display a video
    VideoCapture cap("Video.mp4");
    // VideoCapture cap(0);    //Tell OpenCV to look for webcam when 0
    Mat frame;
    Mat prev_frame;
    Mat frame_gray;
    Mat motion_mask;
    Mat motion_mask_t;
    Mat* Pic = new Mat[1000]; //Array of frames
    int number_N = 15;

    if (!cap.isOpened()) {  //Just check is opened
        return 0;
    }

    for (int i = 0; i < 1000; i++) {
        cap >> frame;       //We take each frame and save it
    
        cvtColor(frame, frame_gray, COLOR_RGB2GRAY);  //Transform into gray scale (when choosing code rgb2gray)

        // if (i > 0) {
        if (i > number_N) {
            //motion_mask is where I store the new differentiated frame
            // absdiff(frame_gray, prev_frame, motion_mask);
            absdiff(frame_gray, Pic[i-number_N], motion_mask);
            //Binary mask (black&white) which removes all pixels under 50
            //while all others above will go to 225
            threshold(motion_mask, motion_mask_t, 50, 225, THRESH_BINARY);
            //imshow("motion_mask", motion_mask);
            imshow("motion_mask", motion_mask_t);
        }

        // frame_gray.copyTo(prev_frame);
        frame_gray.copyTo(Pic[i]);

    imshow("frame", frame);
    waitKey(1);

    }

    delete[] Pic;
    return 0;

}