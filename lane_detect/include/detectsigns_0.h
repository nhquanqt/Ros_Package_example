#ifndef DETECTSIGNS_H
#define DETECTSIGNS_H

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

class DetectSigns{
public:
    DetectSigns();
    ~DetectSigns();

    void update(const Mat &src);

private:
    
};

#endif


