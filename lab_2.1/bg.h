#ifndef BG_H_
#define BG_H_

#include <opencv2/opencv.hpp>

using namespace cv;

//Mat* is a pointer to another matrix called bg
void bg_train(Mat frame, Mat* bg);
void bg_update(Mat frame, Mat* bg);

#endif
