#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


// Good Features to Track + Lukas-Kanade function
int main() {

    //Initialize matrices
    Mat frame, prev_frame, frame_gray, copy;
    VideoCapture cap("Video.mp4");

    //Params for the gff + lk
    vector<Point2f> corners, prev_corners;
    vector<uchar> status;
    vector<float> err;
    //Tell the function how many corners we want to use
    int maxCorners = 100;
    double qualityLevel = 0.01; // 1 is perfetct quality (?)
    double minDistance = 10;    // Number of pixels, minimum between 2 poiunts
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    if(!cap.isOpened()) {
        return 0;
    }

    int step = 100;

    for (int i = 0; i < 1000; i++) {
        cap >> frame;

        // Visualization
        copy = frame.clone();

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        if (i < 5 or i % step == 0) {
            // Select GFF features
            goodFeaturesToTrack(frame_gray, corners, maxCorners, 
                        qualityLevel, minDistance, Mat(), 
                        blockSize, useHarrisDetector, k);
        } else {
            //Tracking
            calcOpticalFlowPyrLK(prev_frame, frame, 
                    prev_corners, corners, status, err);
        }

        //How big the circle we plot
        int r = 4;
        for (int j = 0; j < corners.size(); j++) {
            circle(copy, corners[j], r, Scalar(5*j, 2*j, 255-j), -1, 8, 0);
        }

        // Done on the RGB frame, not gray-scale
        prev_frame = frame.clone();
        prev_corners = corners;

        imshow("Features", copy);
        waitKey(1);

    }

    return 0;
}