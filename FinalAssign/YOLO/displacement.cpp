#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

ofstream result_displ ("displacement.txt");
ofstream average_displ ("average_displacement.txt");


double pointDistance(double const& a, double const& b, double const& c, double const& d);

bool sort_pair(const std::pair<int,int>& a, const std::pair<int,int>& b);

int main(){

    string::size_type sz;

    // - - - - - TRACKING RESULTS - - - - -

    vector<vector<vector<double>>> my_results; // 1st vector is frame, 2nd is person, 3rd is data
    vector<vector<double>> one_frame;
    int cnt = 1;

    ifstream my_file("/home/mmlab/workspace/C++/FinalAssign/YOLO/results/tracking.txt");
    if (my_file.is_open()){
        while (my_file){    // Loop in file
            string s;
            if (!getline(my_file, s)) break;

            istringstream ss(s);
            vector <double> record;

            while (ss) {       // Loop in line
                string s;
                if (!getline(ss,s,',')) break;
                double new_s = stod(s,&sz);
                record.push_back(new_s);
            }

            if (record[0] == cnt){
                one_frame.push_back(record);
            } else {
                my_results.push_back(one_frame);
                one_frame = {};
                one_frame.push_back(record);
                cnt++;
            }

        }

    } else {
        cout << "my_file was not opened" << endl;
    }

    my_file.close();

    // for (int i  = 0; i < my_results.size(); i++){
    //     for (int j = 0; j < my_results[i].size(); j++){
    //         cout << my_results[i][j][0] << " " << my_results[i][j][1]<< " " << my_results[i][j][2] << " " << my_results[i][j][3] << endl;
    //     }
    // }

    // - - - - - GROUND TRUTH - - - - -

    vector<vector<vector<double>>> gt_results;
    one_frame = {};
    cnt = 1;

    ifstream gt_file("/home/mmlab/workspace/C++/FinalAssign/Video/gt/gt.txt");
    if (gt_file.is_open()){
        while (gt_file){
            string s;
            if (!getline(gt_file, s)) break;

            istringstream ss(s);
            vector <double> record;

            while (ss) {
                string s;
                if (!getline(ss,s,',')) break;
                double new_s = stod(s,&sz);
                record.push_back(new_s);
            }

            if (record[0] == cnt){
                one_frame.push_back(record);
            } else {
                gt_results.push_back(one_frame);
                one_frame = {};
                one_frame.push_back(record);
                cnt++;
            }

        }

    } else {
        cout << "gt_file was not opened" << endl;
    }

    gt_file.close();

    // for (int i  = 0; i < gt_results.size(); i++){
    //     for (int j = 0; j < gt_results[i].size(); j++){
    //         cout << gt_results[i][j][0] << " " << gt_results[i][j][1]<< " " << gt_results[i][j][2] << " " << gt_results[i][j][3] << endl;
    //     }
    // }

    // - - - - - Compute displacement - - - - -

    double distance;
    int min_dist, min_idx;
    vector<pair<int,int>> displ_list;


    for (int frame = 0; frame < gt_results.size(); frame++){    // Loop frames
        for (int i = 0; i < gt_results[frame].size(); i++){          // Loop gt_people

            // if (gt_results[frame][i][0] == 9){
            //     for (int j = 0; j < my_results[frame].size(); j++){
            //         if (my_results[frame][j][0] == 1){
            //             distance = pointDistance(gt_results[frame][i][2], gt_results[frame][i][3], my_results[frame][j][2], my_results[frame][j][3]);
            //             displ_list.push_back(make_pair(gt_results[frame][i][0], distance));
            //         }
            //     }
            // }

            if (my_results[frame].size() == 0){
                break;
            }
            min_dist = 100000;
            for (int j = 0; j < my_results[frame].size(); j++){     // Loop my_people
                distance = pointDistance(gt_results[frame][i][2], gt_results[frame][i][3], my_results[frame][j][2], my_results[frame][j][3]);
                if (distance < min_dist){
                    min_dist = distance;
                    min_idx = j;
                }
            }
            displ_list.push_back(make_pair(gt_results[frame][i][1], min_dist));
            my_results[frame].erase(my_results[frame].begin() + min_idx);
        }
    }

    sort(displ_list.begin(), displ_list.end(), sort_pair);

    int ID_cnt = 1;
    int people_cnt = 0;
    int mean_val = 0;

    for (int i = 0; i < displ_list.size(); i++){
        if (get<0>(displ_list[i]) != ID_cnt){
            mean_val /= people_cnt;
            average_displ << "Average displacement for " << ID_cnt << " is: " << mean_val << endl;
            mean_val = 0;
            people_cnt = 0;
            ID_cnt = get<0>(displ_list[i]);
        }
        mean_val += get<1>(displ_list[i]);
        people_cnt++;
        result_displ << get<0>(displ_list[i]) << ", " << get<1>(displ_list[i]) << endl;
        // cout << get<0>(displ_list[i]) << " " << get<1>(displ_list[i]) << endl;
    }

    result_displ.close();
    average_displ.close();

}

// Compute Euclidian distance of 2 points
double pointDistance(double const& a, double const& b, double const& c, double const& d){
    return sqrt(pow(a - c, 2) + pow(b - d, 2));
}

// Sort pairs by first element
bool sort_pair(const std::pair<int,int>& a, const std::pair<int,int>& b){
	return (a.first < b.first);
}