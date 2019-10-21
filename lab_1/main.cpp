#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

using namespace cv;     //To save us time when calling OpenCV

int main() {
    Mat image;
    image = imread("Google.jpg", 1);   //Image reading function, the flag is 0 for grey-scale, 1 for RGB

    imshow("Image window", image);   //Create a new window: give it a name
    waitKey(0); //We use a wait key to give us time to see the image, otherwise shuts off immediately (0 until we press a key)

    return 0;

}