#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>

using namespace cv;
using namespace dnn;
using namespace std;

ofstream result_detec ("../results/detection.txt");
ofstream result_track ("../results/tracking.txt");

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;
vector<Point> centroid_list, prev_centroid_list, unlabeled_bBoxes;
vector<int> ID_list, prev_ID_list;
vector<pair<int, Point>> lost_list;
int ID = 1;

// Threshold for tracking max distance allowed between centroids in 2 consecutive frames
int near_threshold = 50;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out, int& frame_num);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, int& frame_num);

// Compute Euclidian distance of 2 points
double pointDistance(Point const& a, Point const& b);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
{
    // Load names of classes
    string classesFile = "/home/mmlab/workspace/C++/FinalAssign/YOLO/coco.names";
    ifstream ifs(classesFile.c_str());
    string line, output_image;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    String modelConfiguration = "/home/mmlab/workspace/C++/FinalAssign/YOLO/yolov3.cfg";
    String modelWeights = "/home/mmlab/workspace/C++/FinalAssign/YOLO/yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    Mat frame, blob;

    // Process frames
    for (int num = 1; num <= 795; num++)
    {
        String file_name = "/home/mmlab/workspace/C++/FinalAssign/Video/img1/000001.jpg";
        output_image = "/home/mmlab/workspace/C++/FinalAssign/YOLO/img/output_0001.jpg";
        // Loop over images by changing file name
        file_name.replace(file_name.end()-(4+to_string(num).size()), file_name.end()-4, to_string(num));
        output_image.replace(output_image.end()-(4+to_string(num).size()), output_image.end()-4, to_string(num));
        cout << file_name << endl << endl;
        frame = imread(file_name);

        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        
        //Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        // Remove the bounding boxes with low confidence
        postprocess(frame, outs, num);

        // If previous list still has elements, save them in lost_list
        if (prev_centroid_list.size() == prev_ID_list.size() != 0){
            for (int i = 0; i < prev_centroid_list.size(); i++){
                cout << prev_ID_list[i] << " ~ " << prev_centroid_list[i] << endl;
                lost_list.push_back(make_pair(prev_ID_list[i], prev_centroid_list[i]));
            }
        }

        if (num > 1 and unlabeled_bBoxes.size() > 0 and lost_list.size() > 0){
            double dist = 0;
            double min_dist = 1000000;
            int minID = -1;
            int minidx = -1;
            // Assign non-labeled bBoxes to closest lost ID
            for (int i = 0; i < unlabeled_bBoxes.size(); i++){
                for (int j = 0; j < lost_list.size(); j++){
                    dist = pointDistance(unlabeled_bBoxes[i], get<1>(lost_list[j]));
                    if (dist < min_dist and dist < near_threshold+75){
                        min_dist = dist;
                        minID = get<0>(lost_list[j]);
                        minidx = j;
                    }
                }

                if (minID != -1){
                    
                    putText(frame, to_string(minID), unlabeled_bBoxes[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255), 3);
                    centroid_list.push_back(unlabeled_bBoxes[i]);
                    ID_list.push_back(minID);
                    lost_list.erase(lost_list.begin() + minidx);
                    result_track << to_string(num) << ", " << to_string(minID) << ", " << to_string(unlabeled_bBoxes[i].x) << ", " << to_string(unlabeled_bBoxes[i].y) << endl;
                    if (lost_list.size() > 0){
                        break;
                    }
                }

            }

        }

        unlabeled_bBoxes = {};

        // Set current lists as previous ones
        prev_centroid_list = {};
        for (int k = 0; k < centroid_list.size(); k++){
            prev_centroid_list.push_back(centroid_list[k]);
        }
        centroid_list = {};
        prev_ID_list = {};
        for (int k = 0; k < ID_list.size(); k++){
            prev_ID_list.push_back(ID_list[k]);
        }
        ID_list = {};


        string label = format("Image %.i", num);
        putText(frame, label, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
        
        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        imwrite(output_image, detectedFrame);
        
        imshow("Detection & Tracking", frame);
        waitKey(1);
        
    }

    result_detec.close();
    result_track.close();
    
    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, int& frame_num)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold and classes[classIdPoint.x] == "person")
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> tot_bBoxes;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, tot_bBoxes);

    // Call draw function for bounding boxes
    for (size_t i = 0; i < tot_bBoxes.size(); ++i)
    {
        int idx = tot_bBoxes[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, frame_num);
    }
}

// Draw predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, int& frame_num)
{
    // Draw bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 0, 0), 3);

    double distance = 0;
    double min_distance = 1000000;
    int min_ID = -1;
    int min_idx = -1;
    bool has_ID = false;

    // Compute center of bounding box
    Point centroid = Point((left + right)/2, (top + bottom)/2);

    // Write in detection.txt
    result_detec << to_string(frame_num) << ", " << to_string(left) << ", " << to_string(top) << ", " << to_string(right-left) << ", " << to_string(bottom-top) << endl;

    // Calculate closest centroid and give its ID
    if (frame_num == 1){    // If first frame

        putText(frame, to_string(ID), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255), 3);
        result_track << to_string(frame_num) << ", " << to_string(ID) << ", " << to_string(centroid.x) << ", " << to_string(centroid.y) << endl;
        centroid_list.push_back(centroid);
        ID_list.push_back(ID);
        has_ID = true;
        ID++;

    } else {

        // Calculate distance from all previous frame points to get same ID as closest
        for (int i = 0; i < prev_centroid_list.size(); i++){
            distance = pointDistance(centroid, prev_centroid_list[i]);
            // cout << prev_ID_list[i] << " ~ " << distance << endl;
            if (distance < min_distance and distance < near_threshold){
                min_distance = distance;
                min_ID = prev_ID_list[i];
                min_idx = i;
            }
        }

        // Frame width = 768, height = 576
        if (min_ID >= 0){

            putText(frame, to_string(min_ID), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255), 3);
            centroid_list.push_back(centroid);
            ID_list.push_back(min_ID);
            has_ID = true;
            prev_centroid_list.erase(prev_centroid_list.begin() + min_idx);
            prev_ID_list.erase(prev_ID_list.begin() + min_idx);
            result_track << to_string(frame_num) << ", " << to_string(min_ID) << ", " << to_string(centroid.x) << ", " << to_string(centroid.y) << endl;

        } else if (centroid.x < 0 or centroid.y < 100 or centroid.x > 768-150 or centroid.y > 576-150){ // CHANGED
            
            // No close bBox and at the sides
            putText(frame, to_string(ID), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0,225,225), 5);
            result_track << to_string(frame_num) << ", " << to_string(ID) << ", " << to_string(centroid.x) << ", " << to_string(centroid.y) << endl;
            centroid_list.push_back(centroid);
            ID_list.push_back(ID);
            has_ID = true;
            ID++;
        }
    }

    // Keep track of unlabeled bounding boxes
    if (!has_ID){
        unlabeled_bBoxes.push_back(centroid);
    }
    
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

// Compute Euclidian distance of 2 points
double pointDistance(Point const& a, Point const& b){
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}