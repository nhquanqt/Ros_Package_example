#include "detectsigns.h"

void DetectSigns::extractPlane(Mat &src, int n, int ch, Mat &dst)
{
    const int rows = src.rows;
    const int cols = src.cols;
    dst = cv::Mat::zeros(rows, cols, CV_32FC1);

    for (int row = 0; row < rows; row++)
    {
        const float *ptrsrc = src.ptr<float>(n, ch, row);
        float *ptrdst = dst.ptr<float>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrdst[col] = ptrsrc[col];
        }
    }
} 

// Remove the bounding boxes with low confidence using non-maxima suppression
void DetectSigns::postprocess(Mat& frame, const vector<Mat>& outs)
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
            if (confidence > confThreshold)
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
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void DetectSigns::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<String> DetectSigns::getOutputsNames(const Net& net)
{
    cerr<<"CheckName"<<endl;
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

DetectSigns::DetectSigns(){
    confThreshold = 0.5;
    nmsThreshold = 0.4;
    inpWidth = 480;
    inpHeight = 480;

    classesFile = "/home/wan/catkin_ws/src/lane_detect/data/left-right-obj.names";

    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)){
        cerr<<line<<endl;
        classes.push_back(line);
    }

    modelConfiguration = "/home/wan/catkin_ws/src/lane_detect/data/left-right-yolov3-tiny.cfg";
    modelWeights = "/home/wan/catkin_ws/src/lane_detect/data/left-right-yolov3-tiny.backup";

    net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
}
DetectSigns::~DetectSigns(){}

void DetectSigns::update(const Mat &src){
    detect(src);
}

void DetectSigns::detect(const Mat &src){
    Mat blob;
    Mat frame = src.clone();

    // extractPlane(frame,0,0,frame);

    // cerr<<frame.cols<<' '<<frame.rows<<endl;
    // cerr<<frame.size().area()<<endl;
    // cerr<<Size(inpWidth,inpHeight).area()<<endl;


    blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

    net.setInput(blob);

    cerr<<"CheckBlob"<<endl;
    cerr<<blob.size[0]<<' '<<blob.size[1]<<' '<<blob.size[2]<<' '<<blob.size[3]<<' '<<endl;

    cerr<<blob.type()<<endl;
    
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));
    cerr<<"CheckForward"<<endl;
    return;

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);
    
    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time for a frame : %.2f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    
    // Write the frame with the detection boxes
    Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);

    imshow("Signs",frame);
}

void DetectSigns::show(const Mat &src){
    
}