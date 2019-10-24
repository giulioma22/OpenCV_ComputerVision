#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/bgsegm.hpp>

using namespace cv;

//Working on mixtures of Gaussians

int main() {

    Mat frame;
    Mat motion_mask;
    Mat bg;

    Ptr<bgsegm::BackgroundSubtractorMOG> pMOG;
    Ptr<BackgroundSubtractorMOG2> pMOG2;
    int history = 200;  //How many frames are used to compute weights
    int n_mixtures = 200;   //How many Gaussians I will use
    double backgroundRatio = 0.5;   //Ratio btw bg and fg Gaussians
    double learningRate = 0.1;  //How fast update weights
    double noiseSigma = 10;  //For more robustness of brightness changes
    
    pMOG = bgsegm::createBackgroundSubtractorMOG(history, n_mixtures, backgroundRatio, noiseSigma);
    //The bool param is for shadow detection
    pMOG2 = createBackgroundSubtractorMOG2(history, (16.0), true);

    VideoCapture cap("Video.mp4");

    if (!cap.isOpened()) {
        return 0;
    }

for (int i = 0; i < 10000; i++) {
    cap >> frame;
    //To apply MOG to current frame
    //pMOG->apply(frame, motion_mask, learningRate);
    pMOG2->apply(frame, motion_mask, learningRate);
    pMOG2->getBackgroundImage(bg);

    imshow("original", frame);
    imshow("MOG", motion_mask);
    waitKey(1);
}


    return 0;
}