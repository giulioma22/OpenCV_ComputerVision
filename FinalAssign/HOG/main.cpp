#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main( int argc, const char** argv )
{

    //VideoCapture cap(video_location);
    Mat current_frame;

    // Use default pedestrian detector
    //HOGDescriptor hog;
    HOGDescriptor hog(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9);
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    for (int num = 1; num <= 795; num++){
    //for (int num = 1; num <= 1; num++){

        String file_name = "/home/mmlab/workspace/C++/FinalAssign/Video/img1/000001.jpg";
        // Loop over images by changing file name
        file_name.replace(file_name.end()-(4+to_string(num).size()), file_name.end()-4, to_string(num));
        cout << file_name << endl;
        current_frame = imread(file_name);

        // Check if the frame has content
        if(current_frame.empty()){
            cerr << "Video has ended or bad frame was read. Quitting." << endl;
            return 0;
        }

        Mat img = current_frame.clone();
        resize(img,img,Size(img.cols*2, img.rows*2));

        vector<Rect> found;
        vector<double> weights;

        hog.detectMultiScale(img, found, weights);

        // Draw bounding box
        for( size_t i = 0; i < found.size(); i++ )
        {
            Rect r = found[i];
            // Draw and count bounding box only if not too big
            //if (r.width < 150 and r.height < 300){
            if (r.width > 1 and r.height > 1){
                rectangle(img, found[i], cv::Scalar(0,255,0), 3);
            }
        }

        // Show image
        resize(img,img,Size(img.cols/2, img.rows/2));
        imshow("Detection", img);
        //waitKey(0);
        waitKey(1);
    }

    return 0;
}