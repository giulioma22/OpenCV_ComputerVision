#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

String face_cascade_name = "/home/mmlab/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
String face_cascade_name_profile = "/home/mmlab/opencv/data/haarcascades/haarcascade_profileface.xml";

CascadeClassifier face_cascade;
CascadeClassifier face_cascade_profile;

int main() {

    VideoCapture cap(0);
    Mat frame;

    for (int i = 0; i < 1000; i++) {
        cap >> frame;

        // // Load cascades
        if (!face_cascade.load(face_cascade_name)){
            return -1;
        }
        if (!face_cascade_profile.load(face_cascade_name_profile)){
            return -1;
        }

        // Apply classifier to each frame
        vector<Rect> faces;
        vector<Rect> faces_profile;
        Mat frame_gray;

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // Histogram equalization to have better result
        equalizeHist(frame_gray,frame_gray);

        // Detect faces
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 
                            0|CASCADE_SCALE_IMAGE, Size(30,30));
        face_cascade_profile.detectMultiScale(frame_gray, faces_profile, 1.1, 2, 
                            0|CASCADE_SCALE_IMAGE, Size(30,30));

        // Look for ellipses instead of rectangles (closer to shape of face)
        for (int j = 0; j < faces.size(); j++){
            Point center (faces[j].x + faces[j].width*0.5,
                        faces[j].y + faces[j].height*0.5);
            ellipse(frame, center, Size(faces[j].width*0.5, 
                    faces[j].height*0.5),0,0,360,Scalar(0,255,0),4,8,0);
        }

        // For profile faces
        for (int j = 0; j < faces_profile.size(); j++){
            Point center (faces_profile[j].x + faces_profile[j].width*0.5,
                        faces_profile[j].y + faces_profile[j].height*0.5);
            ellipse(frame, center, Size(faces_profile[j].width*0.5, 
                    faces_profile[j].height*0.5),0,0,360,Scalar(0,255,0),4,8,0);
        }

        // Display result
        imshow("Viola-Jones", frame);
        waitKey(1);
    }

    return 0;
}