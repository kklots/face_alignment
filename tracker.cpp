#include "tracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <ctype.h>
#include <fstream>

using namespace cv;
using namespace std;

#define ORIENTATION				 8						
#define CELLSIZE				 12				    	

#define DESCRIPTOR_LENGTH        4*(ORIENTATION*3+1)    
#define TIME_SIZES				 3				    	
#define CONST_H					 0.8					
#define STANDARD_SIZE			 200				   	
#define SAFE_MARGIN              0.5					
#define SWAP(x, y, type) {type tmp = (x); (x) = (y); (y) = (tmp);}

namespace tracker {

#define READ_MATLAB_MATRIX_32(fin, matrix) \
    {                                                   \
        uint32_t rows, cols;                            \
        float minv, step;                               \
        uint32_t *bufMat = NULL;                        \
                                                        \
        if(fin == NULL)                                 \
            return 2;                                   \
                                                        \
        if(fread(&rows, sizeof(uint32_t), 1, fin) != 1) \
                    return 2;                           \
                                                        \
        if(fread(&cols, sizeof(uint32_t), 1, fin) != 1) \
            return 2;                                   \
                                                        \
        assert(rows > 0 && cols > 0);                   \
                                                        \
        bufMat = new uint32_t[rows * cols];             \
                                                        \
        if(fread(&minv, sizeof(float), 1, fin) != 1)    \
            return 2;                                   \
                                                        \
        if(fread(&step, sizeof(float), 1, fin) != 1)    \
            return 2;                                   \
                                                        \
        if(fread(bufMat, sizeof(uint32_t), rows * cols, fin) != rows * cols){ \
            return 2;                                   \
        }                                               \
                                                        \
        matrix.resize(rows, cols);                      \
        float *ptrMat = (float *)matrix.data();         \
        int size = rows * cols;                         \
        for(int i = 0; i < size; i++) \
            ptrMat[i] = bufMat[i] * step + minv;        \
                                                        \
        delete [] bufMat;                               \
    }

#define READ_MATLAB_MATRIX(fin, matrix) \
    {                                                   \
        uint32_t rows, cols;                            \
        float minv, step;                               \
        uint16_t *bufMat = NULL;                        \
                                                        \
        if(fin == NULL)                                 \
            return 2;                                   \
                                                        \
        if(fread(&rows, sizeof(uint32_t), 1, fin) != 1) \
                    return 2;                           \
                                                        \
        if(fread(&cols, sizeof(uint32_t), 1, fin) != 1) \
            return 2;                                   \
                                                        \
        assert(rows > 0 && cols > 0);                   \
                                                        \
        bufMat = new uint16_t[rows * cols];             \
                                                        \
        if(fread(&minv, sizeof(float), 1, fin) != 1)    \
            return 2;                                   \
                                                        \
        if(fread(&step, sizeof(float), 1, fin) != 1)    \
            return 2;                                   \
                                                        \
        if(fread(bufMat, sizeof(uint16_t), rows * cols, fin) != rows * cols){ \
            return 2;                                   \
        }                                               \
                                                        \
        matrix.resize(rows, cols);                      \
        float *ptrMat = (float *)matrix.data();         \
        int size = rows * cols;                         \
        for(int i = 0; i < size; i++) \
            ptrMat[i] = bufMat[i] * step + minv;        \
                                                        \
        delete [] bufMat;                               \
    }


#define READ_MATLAB_MATRIX_W(fin, matrix) \
    {                                                   \
        uint32_t rows, cols;                            \
        float minv, step;                               \
        uint8_t *bufMat = NULL;                         \
                                                        \
        if(fin == NULL)                                 \
            return 2;                                   \
                                                        \
        if(fread(&rows, sizeof(uint32_t), 1, fin) != 1) \
                    return 2;                           \
                                                        \
        if(fread(&cols, sizeof(uint32_t), 1, fin) != 1) \
            return 2;                                   \
                                                        \
        assert(rows > 0 && cols > 0);                   \
                                                        \
        bufMat = new uint8_t[cols];                     \
                                                        \
        matrix.resize(rows, cols);                      \
                                                        \
        for (int y = 0; y < rows; y++) {                \
            if(fread(&minv, sizeof(float), 1, fin) != 1)    \
                return 2;                                   \
                                                            \
            if(fread(&step, sizeof(float), 1, fin) != 1)    \
                return 2;                                   \
                                                            \
            if(fread(bufMat, sizeof(uint8_t), cols, fin) != cols) \
                return 2;                                   \
                                                            \
            for(int x = 0; x < cols; x++)                   \
            matrix(y, x) = minv + bufMat[x] * step;         \
                                                            \
        }                                                   \
        delete [] bufMat;                                   \
    }

void calc_affine(TVec_f &curShape, TVec_f &meanShape, TMat_f &affMat){
    int ptsSize = curShape.rows() / 2;
    TMat_f X(ptsSize * 2, 4);
	TVec_f tempMS(ptsSize*2,1);
    for(int i = 0; i < ptsSize; i++){
        X(i, 0) = curShape[i];
        X(i, 1) = -curShape[i + ptsSize];
        X(i, 2) = 1;
        X(i, 3) = 0;
		tempMS[i]=meanShape[i];
        X(i + ptsSize, 0) = curShape[i + ptsSize];
        X(i + ptsSize, 1) = curShape[i];
        X(i + ptsSize, 2) = 0;
        X(i + ptsSize, 3) = 1;
		tempMS[i+ptsSize]=meanShape[i+ ptsSize];
    }
	
    affMat = X.householderQr().solve(tempMS);
}

void calc_affine(TVec_f &curShape, TVec_f &meanShape, float &angle, float &scale){
    int ptsSize = curShape.rows() / 2;
    TMat_f X(ptsSize * 2, 4);
	TVec_f tempMS(ptsSize*2,1);
    for(int i = 0; i < ptsSize; i++){
        X(i, 0) = curShape[i];
        X(i, 1) = -curShape[i + ptsSize];
        X(i, 2) = 1;
        X(i, 3) = 0;
		tempMS[i]=meanShape[i];
        X(i + ptsSize, 0) = curShape[i + ptsSize];
        X(i + ptsSize, 1) = curShape[i];
        X(i + ptsSize, 2) = 0;
        X(i + ptsSize, 3) = 1;
		tempMS[i+ptsSize]=meanShape[i+ptsSize];
    }
    TMat_f affMat = X.householderQr().solve(tempMS);
    scale = sqrt(affMat(0, 0) * affMat(0, 0) + affMat(1, 0) * affMat(1, 0));
    angle = -atan2(affMat(1, 0), affMat(0, 0)) * 180 / M_PI;
}
void affine_shape(TVec_f &shape, float angle, float scale, cv::Point2f &center)
{
    int ptsSize = shape.rows() / 2;

    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
    TVec_f tempFeats = shape;

    //need verify
    tempFeats.block(0, 0, ptsSize,tempFeats.cols()) = shape.block(0,0,ptsSize,shape.cols()).array() * rot_mat.at<double>(0,0) +
        shape.block(ptsSize,0,ptsSize,shape.cols()).array() * rot_mat.at<double>(0,1) + rot_mat.at<double>(0,2);

    tempFeats.block(ptsSize,0,ptsSize,tempFeats.cols()) = shape.block(0,0,ptsSize,shape.cols()).array() * rot_mat.at<double>(1,0) +
        shape.block(ptsSize,0,ptsSize,shape.cols()).array() * rot_mat.at<double>(1,1) + rot_mat.at<double>(1,2);

    shape = tempFeats;

    float x = center.x;
    float y = center.y;

    center.x = x * rot_mat.at<double>(0, 0) + y * rot_mat.at<double>(0, 1) + rot_mat.at<double>(0, 2);
    center.y = x * rot_mat.at<double>(1, 0) + y * rot_mat.at<double>(1, 1) + rot_mat.at<double>(1, 2);
}

void affine_sample(cv::Mat& src, TVec_f &shape, float angle, float scale, cv::Point2f &center)
{
    int ptsSize = shape.rows() / 2;
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
    cv::warpAffine(src, src, rot_mat, src.size());
    TVec_f tempFeats = shape;

    tempFeats.block(0,0,ptsSize,tempFeats.cols()) = shape.block(0,0,ptsSize,shape.cols()).array() * rot_mat.at<double>(0,0) +
        shape.block(ptsSize,0,ptsSize,shape.cols()).array() * rot_mat.at<double>(0,1) + rot_mat.at<double>(0,2);

    tempFeats.block(ptsSize,0,ptsSize,tempFeats.cols()) = shape.block(0,0,ptsSize,shape.cols()).array() * rot_mat.at<double>(1,0) +
        shape.block(ptsSize,0,ptsSize,shape.cols()).array() * rot_mat.at<double>(1,1) + rot_mat.at<double>(1,2);

    shape = tempFeats;
    float x = center.x;
    float y = center.y;
    center.x = x * rot_mat.at<double>(0, 0) + y * rot_mat.at<double>(0, 1) + rot_mat.at<double>(0, 2);
    center.y = x * rot_mat.at<double>(1, 0) + y * rot_mat.at<double>(1, 1) + rot_mat.at<double>(1, 2);
}

void vertical_mirror(uint8_t *img, int width, int height, int stride){
    int cx = width / 2;

    for(int y = 0; y < height; y++){
        for(int x = 0; x < cx; x++)
            SWAP(img[x], img[width - 1 - x], uint8_t);

        img += stride;
    }
}
void vertical_mirror(TVec_f &shape, int width){
    int ptsSize = shape.size() / 2;
    for(int i = 0; i < ptsSize; i++){
        shape[i] = width - 1 - shape[i];
    }
    for(int i = 0; i < 8; i++){
        SWAP(shape[i], shape[16 - i], float);
        SWAP(shape[i + ptsSize], shape[16 - i + ptsSize], float);
    }

    for(int i = 0; i < 5; i++){
        SWAP(shape[17 + i], shape[26 - i], float);
        SWAP(shape[17 + i + ptsSize], shape[26 - i + ptsSize], float);
    }

    for(int i = 0; i < 4; i++){
        SWAP(shape[36 + i], shape[45 - i], float);
        SWAP(shape[36 + i + ptsSize], shape[45 - i + ptsSize], float);
    }

    SWAP(shape[40], shape[47], float);
    SWAP(shape[40 + ptsSize], shape[47 + ptsSize], float);

    SWAP(shape[41], shape[46], float);
    SWAP(shape[41 + ptsSize], shape[46 + ptsSize], float);

    SWAP(shape[31], shape[35], float);
    SWAP(shape[31 + ptsSize], shape[35 + ptsSize], float);

    SWAP(shape[32], shape[34], float);
    SWAP(shape[32 + ptsSize], shape[34 + ptsSize], float);

    for(int i = 0; i < 3; i++){
        SWAP(shape[48 + i], shape[54 - i], float);
        SWAP(shape[48 + i + ptsSize], shape[54 - i + ptsSize], float);
    }

    for(int i = 0; i < 2; i++){
        SWAP(shape[55 + i], shape[59 - i], float);
        SWAP(shape[55 + i + ptsSize], shape[59 - i + ptsSize], float);

        SWAP(shape[60 + i], shape[64 - i], float);
        SWAP(shape[60 + i + ptsSize], shape[64 - i + ptsSize], float);
    }

    SWAP(shape[65], shape[67], float);
    SWAP(shape[65 + ptsSize], shape[67 + ptsSize], float);

}

void show_image(char* name,cv::Mat &image, vector<float>landmarks,cv::Scalar color){
    int lsize = landmarks.size() / 2;
    for(int i = 0; i < lsize; i++){
		cv::circle(image, cv::Point2f(landmarks[i], landmarks[i + lsize]), 2, cv::Scalar(0,0,255), -1);
    }
	if(lsize==68)
	{
		//outline
		for(int i=0;i<16;i++){
			cv::line(image,cv::Point2f(landmarks[i], landmarks[i + lsize]),cv::Point2f(landmarks[i+1], landmarks[i+1 + lsize]),color,1);
		}
		//eyebrow
		for(int i=17;i<21;i++){
			cv::line(image,cv::Point2f(landmarks[i], landmarks[i + lsize]),cv::Point2f(landmarks[i+1], landmarks[i+1 + lsize]),color,1);
		}
		for(int i=22;i<26;i++){
			cv::line(image,cv::Point2f(landmarks[i], landmarks[i + lsize]),cv::Point2f(landmarks[i+1], landmarks[i+1 + lsize]),color,1);
		}
		//nose
		for(int i=27;i<30;i++){
			cv::line(image,cv::Point2f(landmarks[i], landmarks[i + lsize]),cv::Point2f(landmarks[i+1], landmarks[i+1 + lsize]),color,1);
		}
		cv::line(image,cv::Point2f(landmarks[30], landmarks[30 + lsize]),cv::Point2f(landmarks[33], landmarks[33 + lsize]),color,1);
		for(int i=31;i<35;i++){
			cv::line(image,cv::Point2f(landmarks[i], landmarks[i + lsize]),cv::Point2f(landmarks[i+1], landmarks[i+1 + lsize]),color,1);
		}
		//eyes
		for(int i=36;i<41;i++){
			cv::line(image,cv::Point2f(landmarks[i], landmarks[i + lsize]),cv::Point2f(landmarks[i+1], landmarks[i+1 + lsize]),color,1);
		}
		cv::line(image,cv::Point2f(landmarks[36], landmarks[36 + lsize]),cv::Point2f(landmarks[41], landmarks[41 + lsize]),color,1);
		for(int i=42;i<47;i++){
			cv::line(image,cv::Point2f(landmarks[i], landmarks[i + lsize]),cv::Point2f(landmarks[i+1], landmarks[i+1 + lsize]),color,1);
		}
		cv::line(image,cv::Point2f(landmarks[42], landmarks[42 + lsize]),cv::Point2f(landmarks[47], landmarks[47 + lsize]),color,1);
		//mouth
		for(int i=48;i<59;i++){
			cv::line(image,cv::Point2f(landmarks[i], landmarks[i + lsize]),cv::Point2f(landmarks[i+1], landmarks[i+1 + lsize]),color,1);
		}
		cv::line(image,cv::Point2f(landmarks[48], landmarks[48 + lsize]),cv::Point2f(landmarks[59], landmarks[59 + lsize]),color,1);
		for(int i=60;i<67;i++){
			cv::line(image,cv::Point2f(landmarks[i], landmarks[i + lsize]),cv::Point2f(landmarks[i+1], landmarks[i+1 + lsize]),color,1);
		}
		cv::line(image,cv::Point2f(landmarks[67], landmarks[67 + lsize]),cv::Point2f(landmarks[60], landmarks[60 + lsize]),color,1);
	}
    cv::imshow(name, image);
}

Mat norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(src, dst, 50, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 50, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

Desc_feat::Desc_feat()
{
    _feat = NULL;
    _nCellSize = 0;
    _nNumBins  = 0;
}
Desc_feat::~Desc_feat()
{
    Destroy();
}
void Desc_feat::Init(int cellsize)
{
	_nCellSize = cellsize;
    _nNumBins  = ORIENTATION;
    _feat = (Feat *) feat_new(unsigned long long(_nNumBins));
}
bool Desc_feat::GetVec(float* data,int rows, int cols, int channels, float* vDesc)
{
    feat_put_image(_feat, data, cols, rows, channels, _nCellSize);
    feat_extract(_feat, vDesc);
    return vDesc;
}
void Desc_feat::Destroy()
{
    if(_feat)
    {
        feat_delete(_feat);
    }
    _feat = NULL;
    _nCellSize = 0;
    _nNumBins  = 0;
}


trackerClass::trackerClass(int false_count_times,float is_face_threshold)
{
	for(int i=0;i<MODEL_NUM;i++)
	{
		_tmModels[i].clear();
		_tmValModels[i].clear();
	}
	_curShape.resize(0);
	_ptsSize  = 0;
    _nDescLen = 0;
    _tvDescPlusList.resize(0);
	_cur_model_idx = 0;
	_false_count_times = false_count_times;
	_is_face_threshold = is_face_threshold;
	_descHandl.Init(CELLSIZE);

}

bool trackerClass::load_model(const char *detectionmodel, const char *trackingmodel)
{
	bool result = load_face_detector(detectionmodel);
	if(EXIT_FAILURE == result)
	{
		return EXIT_FAILURE;
	}
	FILE *fin = fopen(trackingmodel, "rb");
	if(fin == NULL){
        return EXIT_FAILURE;
    }

    if(fread(&_stage, sizeof(uint32_t), 1, fin) != 1)
        return EXIT_FAILURE;
    for(int i = 0; i < MODEL_NUM; i++){
        _tmModels[i].resize(_stage);

        for(int j = 0; j < _stage; j++){
            READ_MATLAB_MATRIX(fin, _tmModels[i][j].m);
            READ_MATLAB_MATRIX(fin, _tmModels[i][j].v);
            READ_MATLAB_MATRIX_W(fin, _tmModels[i][j].w);
        }
		READ_MATLAB_MATRIX(fin, _tmMeanShp[i]);
    }

	TMat_f  _pts_3d;
	READ_MATLAB_MATRIX(fin, _pts_3d);
	_curShape = TVec_f(_tmMeanShp[_cur_model_idx].size(),1);
	_ptsSize = _tmMeanShp[_cur_model_idx].size()/2;
	for(int i=0;i<_ptsSize;i++)
	{
		TVec_f x_3d = _pts_3d.block(0,i,1,1);
		TVec_f y_3d = _pts_3d.block(1,i,1,1);
		TVec_f z_3d = _pts_3d.block(2,i,1,1);
		_model_points.push_back(Point3d(x_3d[0],y_3d[0],z_3d[0]));
	}

    for(int i = 0; i < MODEL_NUM; i++){
		_tmValModels[i].resize(1);
		READ_MATLAB_MATRIX(fin, _tmValModels[i][0].m);
        READ_MATLAB_MATRIX(fin, _tmValModels[i][0].v);
        READ_MATLAB_MATRIX_W(fin, _tmValModels[i][0].w);
	}

    fclose(fin);
	for(int j=0;j<MODEL_NUM;j++)
	{
		TVec_f meanShape = TVec_f(_tmMeanShp[j].size(),1);	
		meanShape.block(0,0,_tmMeanShp[j].size()/2,1)=_tmMeanShp[j].block(0,0,_tmMeanShp[j].size()/2,1);
		meanShape.block(_tmMeanShp[j].size()/2,0,_tmMeanShp[j].size()/2,1)=_tmMeanShp[j].block(0,1,_tmMeanShp[j].size()/2,1);
		float mean_x=0,mean_y=0;
		for(int i=0;i<_ptsSize;i++)
		{
			mean_x += meanShape[i];
			mean_y += meanShape[i+_ptsSize];
		}
		mean_x/=_ptsSize;
		mean_y/=_ptsSize;
		_tmCenter[j] = cv::Point2f(mean_x,mean_y);
	}
    if(_ptsSize <= 0)
        return EXIT_FAILURE;
    _nDescLen = DESCRIPTOR_LENGTH;
    _tvDescPlusList = TVec_f(_nDescLen * _ptsSize + 1 );
	_tvDescPlusList.setZero();
    _tvDescPlusList[_nDescLen * _ptsSize] = 1;
	return EXIT_SUCCESS;
}

bool trackerClass::load_face_detector(const char* cascadePath)
{
    if(_fd.load(cascadePath)){
        cout << "Error loading face detection model." << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int  trackerClass::detect_face(cv::Mat &img)
{
    std::vector<detector::Rect> rects;
    std::vector<detector::Shape> shapes;
    int fsize = 0;
	Mat gray;
	if(img.channels()==3)
	{
		cvtColor(img,gray,CV_BGR2GRAY);
	}else
	{
		gray=img;
	}
    _fd.detect(gray.data, gray.cols, gray.rows, gray.step, rects, shapes);
    fsize = rects.size();

    if(fsize > 0)
    {
        int maxWidth=0;
        int idx;

        for(int i = 0; i < fsize; i++){
            if(maxWidth < rects[i].width){
                maxWidth = rects[i].width;
                idx=i;
            }
        }
		for(int i = 0; i < _ptsSize; i++){
			_curShape[i] = shapes[idx][i].x;
			_curShape[i + _ptsSize] = shapes[idx][i].y;
		}

	}

    return (fsize > 0);
}
void trackerClass::GetPose(double* pose)
{
	for(int i=0;i<6;i++)
		pose[i] = _params[i];
	return;
}
bool trackerClass::Desec_Get(cv::Mat inImg, float fx, float fy, float *vDesc){
	int patchWidthHalf = CELLSIZE;
	int x = cvFloor(fx);
	int y = cvFloor(fy);
	cv::Mat roiImg;
	if (x - patchWidthHalf < 0 || y - patchWidthHalf < 0 || x + patchWidthHalf >= inImg.cols-1 || y + patchWidthHalf >= inImg.rows-1) {
		// The feature extraction location is too far near a border. We extend the
		// image (add a black canvas) and then extract from this larger image.
		int borderLeft = (x - patchWidthHalf) < 0 ? std::abs(x - patchWidthHalf) : 0; // x and y are patch-centers
		int borderTop = (y - patchWidthHalf) < 0 ? std::abs(y - patchWidthHalf) : 0;
		int borderRight = (x + patchWidthHalf) >= inImg.cols-1 ? std::abs(inImg.cols-1 - (x + patchWidthHalf)) : 0;
		int borderBottom = (y + patchWidthHalf) >= inImg.rows-1 ? std::abs(inImg.rows-1 - (y + patchWidthHalf)) : 0;

		cv::Mat extendedImage = inImg.clone();
		cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
		cv::Rect roi((x - patchWidthHalf) + borderLeft, (y - patchWidthHalf) + borderTop, patchWidthHalf * 2, patchWidthHalf * 2); // Rect: x y w h. x and y are top-left corner.
		roiImg = extendedImage(roi).clone(); // clone because we need a continuous memory block
	}
	else{
		cv::Rect roi(x - patchWidthHalf, y - patchWidthHalf, patchWidthHalf * 2, patchWidthHalf * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
		roiImg = inImg(roi).clone(); // clone because we need a continuous memory block
	}

	if(3==roiImg.channels())
	{
		float* data = new float[roiImg.rows*roiImg.cols*roiImg.channels()];
		int nl= roiImg.rows; // number of lines  
		int nc= roiImg.cols; // number of columns  
		int count = 0;	
		for(int k=roiImg.channels()-1;k>=0;k--)
		{
			for (int j=0; j<nl; j++) 
			{
				for (int i=0; i<nc; i++)
				{ 
					data[count] = float(roiImg.at<cv::Vec3b>(j,i)[k]);
					count++;
				}                 
			}
		}
		_descHandl.GetVec((float*)data,roiImg.rows,roiImg.cols,roiImg.channels(),vDesc);
		delete [] data;
	}else
	{
		roiImg.convertTo(roiImg,CV_32F);
		_descHandl.GetVec((float*)roiImg.data,roiImg.rows,roiImg.cols,roiImg.channels(),vDesc);
	}
	roiImg.release();
	return true;
}
bool trackerClass::Desec_GetAll(cv::Mat inImg,int curStage){

	float fx,fy;
	for(int i = 0; i < _ptsSize; i++){
		fx = _curShape[i];
		fy = _curShape[i+_ptsSize];
		int mode = 0;
		if(!Desec_Get(inImg, fx, fy, &_tvDescPlusList[i*_nDescLen]))
			return false;
	}
	TVec_f feature =  _tvDescPlusList.block(0,0,_ptsSize*_nDescLen,1);
	float meanValue = feature.mean();
	_tvDescPlusList.block(0,0,_ptsSize*_nDescLen,1)-=meanValue*feature.setOnes();
	double totalnorm = _tvDescPlusList.block(0,0,_ptsSize*_nDescLen,1).dot(_tvDescPlusList.block(0,0,_ptsSize*_nDescLen,1));
	_tvDescPlusList.block(0,0,_ptsSize*_nDescLen,1)=_tvDescPlusList.block(0,0,_ptsSize*_nDescLen,1)*sqrt((_ptsSize*_nDescLen-1)/totalnorm);
	return true;
}
bool trackerClass::TrackFeats2D(cv::Mat &src,long count,bool &isTrackingSuccess){
	float yaw_angles = _params[1]*57.29578;
	if(yaw_angles>20)
	{
		_cur_model_idx = 1;
	}
	else if(yaw_angles<-20)
	{
		_cur_model_idx = 2;
	}else 
	{
		_cur_model_idx = 0;
	}

	static float angle;
	static float scale;
    cv::Point2f shp_center;
	cv::Point2f mean_center;

	TVec_f meanShape = TVec_f(_ptsSize*2,1);	
	{ 
		TVec_f meanShape_portion;
		TVec_f _curShape_portion;
		meanShape.block(0,0,_ptsSize,1)=_tmMeanShp[_cur_model_idx].block(0,0,_ptsSize,1);
		meanShape.block(_ptsSize,0,_ptsSize,1)=_tmMeanShp[_cur_model_idx].block(0,1,_ptsSize,1);
		meanShape_portion = TVec_f(_ptsSize*2,1);
		_curShape_portion = TVec_f(_ptsSize*2,1);
		_curShape_portion.block(0,0,_ptsSize,1) = _curShape.block(0,0,_ptsSize,1);
		_curShape_portion.block(_ptsSize,0,_ptsSize,1) = _curShape.block(_ptsSize,0,_ptsSize,1);
		meanShape_portion.block(0,0,_ptsSize,1) = meanShape.block(0,0,_ptsSize,1);
		meanShape_portion.block(_ptsSize,0,_ptsSize,1) = meanShape.block(_ptsSize,0,_ptsSize,1);	
		calc_affine(_curShape_portion, meanShape_portion, angle, scale);
	} 
	float mean_x=0,mean_y=0;
	for(int i=0;i<_ptsSize;i++)
	{
		mean_x += _curShape[i];
		mean_y += _curShape[i+_ptsSize];
	}
	shp_center = cv::Point2f(mean_x/_ptsSize,mean_y/_ptsSize);
	affine_sample(src, _curShape, angle, scale, shp_center);
	mean_center = _tmCenter[_cur_model_idx];
	meanShape.block(0,0,_ptsSize,1)=_tmMeanShp[_cur_model_idx].block(0,0,_ptsSize,1);
	meanShape.block(_ptsSize,0,_ptsSize,1)=_tmMeanShp[_cur_model_idx].block(0,1,_ptsSize,1);
	vector <Mat> imgs;
	imgs.resize(3);
	imgs[2]=src.clone();
	cv::pyrDown(src,imgs[1]);
	cv::pyrDown(imgs[1],imgs[0]);
	int imgIdx = 0;
	TVec_f DescList = TVec_f(68*_nDescLen+1);
	DescList[68*_nDescLen]=1;
	for(int nIterID=0;nIterID<_stage;nIterID++)
	{
		if(nIterID==0)
		{
			for(int i=0;i<_ptsSize;i++)
			{
				_curShape[i]=(meanShape[i]*0.95+shp_center.x-mean_center.x)*0.25;
				_curShape[i+_ptsSize]=(meanShape[i+_ptsSize]*0.95+shp_center.y-mean_center.y)*0.25;
			}
			imgIdx=0;
		}else if(nIterID==1)
		{
			_curShape=_curShape*2;
			imgIdx=1;
		}else if(nIterID==2)
		{
			_curShape=_curShape*2;
			imgIdx=2;
		}
		if(!Desec_GetAll(imgs[imgIdx],nIterID))
				return false;
		if(nIterID==_stage-1)
		{
			DescList = _tvDescPlusList;
		}
		_tvDescPlusList.block(0, 0, _ptsSize * _nDescLen, _tvDescPlusList.cols()) -= _tmModels[_cur_model_idx][nIterID].m;
		_tvDescPlusList.block(0, 0, _ptsSize * _nDescLen, _tvDescPlusList.cols())  = _tvDescPlusList.block(0, 0, _ptsSize * _nDescLen,_tvDescPlusList.cols()).array() / _tmModels[_cur_model_idx][nIterID].v.array();
		_curShape += (_tvDescPlusList.transpose() * _tmModels[_cur_model_idx][nIterID].w).transpose();
	}
	if(0==imgIdx)
		_curShape*=4;
	else if(1==imgIdx)
		_curShape*=2;
	DescList.block(0, 0, 68 * _nDescLen, _tvDescPlusList.cols()) -= _tmValModels[_cur_model_idx][0].m;
	DescList.block(0, 0, 68 * _nDescLen, _tvDescPlusList.cols())  = DescList.block(0, 0, 68 * _nDescLen,_tvDescPlusList.cols()).array() / _tmValModels[_cur_model_idx][0].v.array();
	TVec_f isFace = (DescList.transpose() * _tmValModels[_cur_model_idx][0].w).transpose();
	affine_shape(_curShape, -angle, 1 / scale, shp_center);
	if(abs(isFace[0])<this->_is_face_threshold)
		isTrackingSuccess = true;
	else
		isTrackingSuccess = false;
	
	return true;
}
bool trackerClass::Track2D(cv::Mat &src, std::vector<float>& face_pts)
{
	static long count = 0;
    static bool redetection = true;
	static int  false_count = 0;
	static TVec_f T[TIME_SIZES];
	static float mean_x[TIME_SIZES];
	static float mean_y[TIME_SIZES];
	static int continous_error = 0;
	static bool isTrackingSuccess = false;
	Mat gray;
	if(src.channels()==3)
	{
		cvtColor(src,gray,CV_BGR2GRAY);
	}else
	{
		gray=src.clone();
	}
	if(redetection)
	{
		int isdetectioned = detect_face(gray);
		if(!isdetectioned)
		{
			_cur_model_idx = 0;
			count=0;
			return false;
		}
		else
		{
			redetection = false;
		}
	}
	float minx = _curShape[0] , miny = _curShape[_ptsSize] , maxx = _curShape[0] , maxy = _curShape[_ptsSize];
    for(int i = 1; i < _ptsSize; i++){
        minx = _MIN(minx, _curShape[i]);
        maxx = _MAX(maxx, _curShape[i]);
        miny = _MIN(miny, _curShape[i + _ptsSize]);
        maxy = _MAX(maxy, _curShape[i + _ptsSize]);
    }
	float max_length = max(sqrt(pow(_curShape[27]-_curShape[57],2)+pow(_curShape[95]-_curShape[125],2)),sqrt(pow(_curShape[36]-_curShape[45],2)+pow(_curShape[104]-_curShape[113],2)));
	int safe_edge = max_length*SAFE_MARGIN;
	minx = _MAX(minx-safe_edge,0);
	miny = _MAX(miny-safe_edge,0);
	maxx = _MIN(maxx+safe_edge,gray.cols-1);
	maxy = _MIN(maxy+safe_edge,gray.rows-1);
	if(maxx-minx+1<=20||maxy-miny+1<=20) 
	{
		redetection = true;
		count=0;
		return false;
	}	

	float resize_scale = max_length/STANDARD_SIZE;
	
	Mat src_crop = gray(cv::Rect(minx,miny,maxx-minx+1,maxy-miny+1));
	
	for(int i = 0; i < _ptsSize; i++){
		_curShape[i]-=minx;
		_curShape[i+_ptsSize]-=miny;
	}
	
	{
		resize(src_crop,src_crop,cv::Size(src_crop.cols/resize_scale,src_crop.rows/resize_scale));
		for(int i = 0; i < _ptsSize; i++){
		_curShape[i]/=resize_scale;
		_curShape[i+_ptsSize]/=resize_scale;
		}
	}

	TrackFeats2D(src_crop,count,isTrackingSuccess);
	if(false==isTrackingSuccess)
	{
		continous_error++;
	}else
	{
		continous_error=0;
	}


	for(int i = 0; i < _ptsSize; i++){
		_curShape[i]*=resize_scale;
		_curShape[i+_ptsSize]*=resize_scale;
	}
	
	for(int i = 0; i < _ptsSize; i++){
		_curShape[i]+=minx;
		_curShape[i+_ptsSize]+=miny;
	}


	if(count<TIME_SIZES-1)
	{
		T[count] = _curShape;
		count++;
	}else
	{	
		T[TIME_SIZES-1] = _curShape;
		for(int ind_pt = 0;ind_pt<_ptsSize;ind_pt++)
		{
			float weights[TIME_SIZES];
			float sum_weights=0;
			float sum_x=0;
			float sum_y=0;
			float max_dist = sqrt(std::pow(_curShape[ind_pt]-T[0][ind_pt],2)+std::pow(_curShape[ind_pt+_ptsSize]-T[0][ind_pt+_ptsSize],2));
			float max_idx = 0;
			for(int ind_time = 1 ; ind_time  <TIME_SIZES-1 ; ind_time++)
			{
				float dist = sqrt(std::pow(_curShape[ind_pt]-T[ind_time][ind_pt],2)+std::pow(_curShape[ind_pt+_ptsSize]-T[ind_time][ind_pt+_ptsSize],2));
				if(max_dist<dist)
				{
					max_dist=dist;
					max_idx=ind_time;
				}
			}
			for(int ind_time =0 ; ind_time  <TIME_SIZES ; ind_time++)
			{
				weights[ind_time] = std::exp(-((TIME_SIZES-1)-ind_time)*CONST_H*max_dist);
				sum_weights+=weights[ind_time];
				sum_x+=weights[ind_time]*T[ind_time][ind_pt];
				sum_y+=weights[ind_time]*T[ind_time][ind_pt+_ptsSize];
			}
			_curShape[ind_pt]=sum_x/sum_weights;
			_curShape[ind_pt+_ptsSize]=sum_y/sum_weights;
			
		}
		T[TIME_SIZES-1] = _curShape;

		for(int ind_time =0 ; ind_time  <TIME_SIZES-1 ; ind_time++)
		{
			T[ind_time] = T[ind_time+1];	
		}
		count++;
	}
	if(continous_error>this->_false_count_times)
	{
		redetection = true;
		return false;
	}
	else
	{
		static Vec3d vec_trans;
		static Vec3d vec_rot;
		TVec_f _curShape3D = _curShape;	
		double fPixFocusL = src.cols;
		double rows = src.rows;
		double cols = src.cols;
		Matx33d camera_matrix(fPixFocusL, 0, src.cols/2, 0, fPixFocusL, src.rows/2, 0, 0, 1);
		if(count<TIME_SIZES-1)
		{
			vec_rot(0)   =  0;
			vec_rot(1)   =  0;
			vec_rot(2)   =  0;
			vec_trans(0) =  0;
			vec_trans(1) =  0;
			vec_trans(2) =  300;
		}
		Mat_<double> landmarks_2D(51, 2);
		Mat_<double> landmarks_3D(51, 3);
		for(int i=17;i<_ptsSize;i++)
		{
			landmarks_2D(i-17,0) = _curShape[i];
			landmarks_2D(i-17,1) = _curShape[i+68];
			landmarks_3D(i-17,0) = _model_points[i].x;
			landmarks_3D(i-17,1) = _model_points[i].y;
			landmarks_3D(i-17,2) = _model_points[i].z;
		}
		solvePnP(landmarks_3D, landmarks_2D, camera_matrix, Mat(), vec_rot, vec_trans, true);
		for(int i=0;i<3;i++)
		{	
			_params[i]    =  vec_rot[i];
			_params[i+3]  =  vec_trans[i];

		}
		std::memcpy(&face_pts[0], _curShape.data(), 2*_ptsSize*sizeof(float));
		redetection = false;
		return true;
	}
}
}

