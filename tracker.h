#ifndef _TRACKER_
#define _TRACKER_

#include "Def_type.h"
#include "joint_face.h"
#include "feat.h"
#include <vector>

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
        bool  load_model(const char *detectionmodel, const char *trackingmodel);	  //载入模型
		bool  Track2D(cv::Mat &src, std::vector<float>& face_pts);
		inline int GetPtsNum(){return _ptsSize;}
		void GetPose(double* pose);
		vector<cv::Point3d>       _model_points;                                      //三维空间点   
	private:
		bool  load_face_detector(const char* cascadePath);
		int   detect_face(cv::Mat &inputImage);
		bool  TrackFeats2D(cv::Mat &src,long count,bool &isTrackingSuccess);
		bool  Desec_Get(cv::Mat inImg, float fx, float fy, float *vDesc);			  //获得指定位置局部描述
		bool  Desec_GetAll(cv::Mat inImg,int curStage);	                              //获得所有特征
	private:
		detector::Cascade              _fd;                                           //人脸识别器
		std::vector<Model>        _tmModels[MODEL_NUM];                               //特征模型
		std::vector<Model>        _tmValModels[MODEL_NUM];                            //人脸验证模型
		TMat_f                    _tmMeanShp[MODEL_NUM];                              //形状均值
		cv::Point2f               _tmCenter[MODEL_NUM];                               //中心点
		uint32_t                  _stage;                                             //迭代次数
		TVec_f                    _curShape;                                          //当前形状
		int                       _cur_model_idx;                                     //当前model序号
		int                       _ptsSize;                                           //点的数量
		int                       _nDescLen;                                          //特征长度
		TVec_f                    _tvDescPlusList;                                    //特征向量
		Desc_feat                 _descHandl;				                          //HOG特征提取器
		int						  _false_count_times;                                 //允许连续追踪失败的帧总数
		float                     _is_face_threshold;                                 //判断是否是人脸的分割阈值
		double                    _params[6];
	};
	void  show_image(char* name,cv::Mat &image, vector<float>landmarks, cv::Scalar color = cv::Scalar(0, 255, 0));
}


#endif
