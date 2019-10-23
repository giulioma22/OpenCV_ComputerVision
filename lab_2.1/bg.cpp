#include "bg.h"    //Just including the header since it has everything

static int ctr = 1;

void bg_train(Mat frame, Mat* bg) {
    if (ctr == 1) {
        //Initial BG storage
        frame.copyTo(*bg);
    }
    ctr++;

}

void bg_update(Mat frame, Mat* bg) {
    float alpha = 0.01; //Closer to 0, slower to update bg
    *bg = alpha*frame + *bg*(1 -alpha);
}