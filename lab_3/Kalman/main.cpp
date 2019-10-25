#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


struct mouse_info_struct { int x,y; };
struct mouse_info_struct mouse_info = {-1,-1}, last_mouse;

vector<Point> mousev,kalmanv;

void on_mouse(int event, int x, int y, int flags, void* param) {
	{
		last_mouse = mouse_info;
		mouse_info.x = x;
		mouse_info.y = y;

	}
}

// plot points
#define drawCross( center, color, d )                                 \
line( img, Point( center.x - d, center.y - d ),                \
Point( center.x + d, center.y + d ), color, 2, CV_8U, 0); \
line( img, Point( center.x + d, center.y - d ),                \
Point( center.x - d, center.y + d ), color, 2, CV_8U, 0 )

int main () {
    
    Mat img(1000, 1000, CV_8UC3);

    //Size of the (x,y,v_x,v_y), size measurement (x,y) of the mouse
    KalmanFilter KF(4, 2, 0);
    Mat_<float> state(4,1); // x,y,v_x,v_y
    Mat processNoise(4,1,CV_32F);
    Mat_<float> measurement(2,1);   // x,y measured position of the mouse
    measurement.setTo(0);   // Initial measurement for the KF

    //Creating the window and allow interaction with our mouse
    namedWindow("Mouse Kalman");
    setMouseCallback("Mouse Kalman", on_mouse, 0);

    KF.statePre.at<float>(0) = mouse_info.x;
    KF.statePre.at<float>(1) = mouse_info.y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;

    KF.transitionMatrix = (Mat_<float> (4,4) << 
                            1,0,1,0,
                            0,1,0,1,
                            0,0,1,0,
                            0,0,0,1);

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(0.1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    // Vectors for visualization
    mousev.clear();
    kalmanv.clear();

    for(;;) {
        // Prediction, to update the statePre variable
        Mat prediction = KF.predict();  // (2,1) matrix
        Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

        // Measurement
        measurement(0) = mouse_info.x;
        measurement(1) = mouse_info.y;

        Point measPt(measurement(0), measurement(1));
        mousev.push_back(measPt);

        // Update phase (Kalman gain + weights)
        Mat estimated = KF.correct(measurement);
        Point statePt(estimated.at<float>(0),estimated.at<float>(1));
        kalmanv.push_back(statePt);

        // Plot and visualization
        img = Scalar::all(0);   // Settimg images to black (0) at each iter
        
        drawCross(statePt, Scalar(255,255,255), 5); // White
        drawCross(measPt, Scalar(0,0,255), 5);      // Red
        drawCross(predictPt, Scalar(0,255,0), 5);   // Green

        for (int i = 0; i < mousev.size() -1; i++) {
            line(img, mousev[i], mousev[i+1], Scalar(255,255,0), 1);   // Cyan
        }

        for (int i = 0; i < kalmanv.size() -1; i++) {
            line(img, kalmanv[i], kalmanv[i+1], Scalar(0,255,0), 1);   // Green
        }

        imshow("Mouse Kalman", img);
        waitKey(100);


    }


    return 0;
}