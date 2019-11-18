#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>  // SIFT function

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(){

    // Loading the images
    Mat img_object = imread("box.png", 0);  // 0 for having grayscale image
    Mat img_scene = imread("box_in_scene.png", 0);

    // Step 1: Detect keypoints with SIFT, then compute descriptor (i.e. strength of feature)
    Ptr<SIFT> detector = SIFT::create(400); // 400 = min quality level for a feature to be detected
    vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;

    // Compute SIFT
    detector -> detectAndCompute(img_object, noArray(), keypoints_object, descriptors_object);
    detector -> detectAndCompute(img_scene, noArray(), keypoints_scene, descriptors_scene);

    // Display
    Mat keyPlot1, keyPlot2;

    drawKeypoints(img_object, keypoints_object, keyPlot1, Scalar(0,0,255), 
                DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imshow("Object", keyPlot1);

    drawKeypoints(img_scene, keypoints_scene, keyPlot2, Scalar(0,0,255), 
                DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imshow("Scene", keyPlot2);

    // Note: The images that we get will have some "unusual" key features (e.g. plain white areas)
    // because they remain constant at different resolution levels, even though they don't represent
    // a strong feature (e.g. a border, a point)

    waitKey(0);     // Img shown until we press a key


    // Step 2: Matching descriptors btw the 2 images
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;

    matcher.match(descriptors_object, descriptors_scene, matches);
    
    Mat img_matches;

    // Drawing the matches between the 2 images
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, 
                matches, img_matches, Scalar::all(-1), Scalar::all(-1), 
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Note: By plotting now w/o filtering, we have a ton of (mis)matches

    // Get only good matches
    vector<DMatch> good_matches;

    // Filtering on the matches (thresholdd of distance 150)
    for (int i = 0; i < descriptors_object.rows; i++){
        if (matches[i].distance < 150){
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_matches_good;

    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, 
                good_matches, img_matches_good, Scalar::all(-1), Scalar::all(-1), 
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    imshow("Good matches", img_matches_good);
    waitKey(0);

    // Step 3: Stitching, so distort the image to apply it to the scene
    Mat result;

    // Localize the object
    vector<Point2f> obj, scene;
    for (int i = 0; i < good_matches.size(); i++){
        // Get keypoints from good_matches
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

    // Find homography matrix that correctly transforms the image (e.g. traslation, rotation, etc.)
    Mat H = findHomography(obj, scene, RANSAC);
    // Apply transformation to the image
    warpPerspective(img_object, result, H, Size(img_scene.cols, img_scene.rows), INTER_CUBIC);

    // Create black image
    Mat result_mask = Mat::zeros(result.size(), CV_8UC1);
    result_mask.setTo(255, result!=0);

    imshow("Mask", result_mask);
    waitKey(0);

    result.copyTo(img_scene, result_mask);
    imshow("Stitched", img_scene);
    waitKey(0);


    return 0;
}