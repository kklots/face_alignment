#include "tracker.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <ctime>
using namespace std;
using namespace cv;

int main(int argc, char **argv)
{	
	char detectionmodel[] = "./models/detectionmodel.bin";
	char trackingmodel[]  = "./models/trackingmodel.bin";
	tracker::trackerClass mytracker(3,0.3);
	bool isFailed = mytracker.load_model(detectionmodel,trackingmodel);
	if(isFailed) return EXIT_FAILURE;
	bool control = true;
	cv::VideoCapture cap;

	if(*argv[1]=='0')
	{
		cap.open(0);
	}else
	{
		cap.open(argv[1]);
	}

	cv::Mat frame;
    clock_t startTime, endTime;
	std::vector<float> face_pts(mytracker.GetPtsNum()*2);
	char text[100];
	clock_t iters_times;
	float fps = 0;
	double pose[6];
	while (control)
	{		
		cap >> frame;
		if (frame.empty())
			break;
		startTime = clock();
		bool isTracking = mytracker.Track2D(frame, face_pts);//regression	
		mytracker.GetPose(pose);	
		endTime = clock();
		iters_times = max(int(endTime - startTime),1);		
		fps = float(CLOCKS_PER_SEC)/float(iters_times);
		sprintf(text,"fps: %.2f",fps);
		cv::putText(frame,text,cv::Point(10,20),1,1,cv::Scalar(0,255,0),2);
		sprintf(text,"pitch: %.1f , yaw: %.1f , roll: %.1f", pose[0]/(2*M_PI)*360,pose[1]/(2*M_PI)*360,pose[2]/(2*M_PI)*360);
		cv::putText(frame,text,cv::Point(10,50),1,1,cv::Scalar(0,255,0),2);
		if(isTracking)
			tracker::show_image("test",frame, face_pts);
		else
			imshow("test",frame);
		if(waitKey(1) == 27)//esc exit
			control = false;
	}
	return EXIT_SUCCESS;
}