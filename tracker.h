/* facial keypoints tracker, implemented by Lixuan. */
/* email: 15829923@qq.com   kklotss@gmail.com */

#ifndef _TRACKER_
#define _TRACKER_

#include "Def.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "feat.h"
#include <vector>
#include <string>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <stdint.h>


using namespace std;
#define MODEL_NUM 3

namespace tracker
{		
	class Desc_feat
	{
		public:
			Desc_feat();
			~Desc_feat();
			void Init(int cellsize);
			bool GetVec(float* data,int rows, int cols, int channels, float* vDesc);
			void Destroy();	
		private:
			Feat*      _feat;
			int        _nCellSize,_nNumBins;
	};

	typedef struct Model_t
	{
		TMat_f m;
		TMat_f v;
		TMat_f w;
	} Model;
	
	class trackerClass
	{

	public:
		trackerClass(int false_count_times,float is_face_threshold);										                                               
		~trackerClass(){};
        bool  load_model(const char *detectionmodel, const char *trackingmodel);	  
		bool  Track2D(cv::Mat &src, std::vector<float>& face_pts);
		inline int GetPtsNum(){return _ptsSize;}
		void GetPose(double* pose);
		vector<cv::Point3d>       _model_points;                                     
	private:
		bool  load_face_detector(const char* cascadePath);
		int   detect_face(cv::Mat &inputImage);
		bool  TrackFeats2D(cv::Mat &src,long count,bool &isTrackingSuccess);
		bool  Desec_Get(cv::Mat inImg, float fx, float fy, float *vDesc);			 
		bool  Desec_GetAll(cv::Mat inImg,int curStage);	                              
	private:
		CascadeClassifier         _fd;                                           
		std::vector<Model>        _tmModels[MODEL_NUM];                              
		std::vector<Model>        _tmValModels[MODEL_NUM];                          
		TMat_f                    _tmMeanShp[MODEL_NUM];                              
		cv::Point2f               _tmCenter[MODEL_NUM];                              
		uint32_t                  _stage;                                            
		TVec_f                    _curShape;                                        
		int                       _cur_model_idx;                                   
		int                       _ptsSize;                                          
		int                       _nDescLen;                                       
		TVec_f                    _tvDescPlusList;                               
		Desc_feat                 _descHandl;				                    
		int						  _false_count_times;                              
		float                     _is_face_threshold;                              
		double                    _params[6];
	};
	void  show_image(char* name,cv::Mat &image, vector<float>landmarks, cv::Scalar color = cv::Scalar(0, 255, 0));
}


#endif
