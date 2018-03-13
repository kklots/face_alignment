/****************************************
 * Project: Joint detect
 * File: joint_face.h
 * Author: Hu Neng
 * Date: 2015/10/19
 *
 * Copyright 2015, Top+ vision, inc
 ****************************************/

#ifndef _JOINT_FACE_H_
#define _JOINT_FACE_H_

#include <vector>
#include <string>
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <string.h>
#include <float.h>
#include <time.h>
#include <stdint.h>
#include <string.h>


namespace detector{
#define HU_MIN(i, j) (((i) > (j)) ? (j) : (i))
#define HU_MAX(i, j) ((i) < (j) ? (j) : (i))

typedef struct {
    float x;
    float y;
} Point2f;


typedef struct {
    int x, y;
    int width, height;
} Rect;

typedef struct{
    int width, height;
} HSize;


typedef std::vector<Point2f> Shape;


class BoostCart;
/******************* Face Detector ******************/
//Usage:
//1) Create object of Cascade
//2) Load model
//3) Using fuction detect to detect face on image

class Cascade{
public:
    Cascade(){}
    ~Cascade();

    void detect(uint8_t *img, int width, int height, int stride, std::vector<Rect> &rects, std::vector<Shape > &shapes,
            float startScale = 0.0f, float endScale = 0.0f, int layers = 0, float offsetFactor = 0.2f);
    void detect(uint8_t *img, int width, int height, int stride, std::vector<Rect> &rects, std::vector<Shape > &shapes, std::vector<float> &confs,
            float startScale = 0.0f, float endScale = 0.0f, int layers = 0, float offsetFactor = 0.2f);

    int validate(uint8_t *img, int width, int height, int stride);

    int load(const char *fileName);
    HSize get_windows_size(){return winSize_;}

private:
    void detect_patch(uint8_t *img, int width, int height, int stride, Rect &rect, int dx, int dy, float scale, std::vector<Rect> &rects, std::vector<Shape> &shapes, std::vector<float> &confs);
    std::vector<BoostCart *> boosts_;
    Shape meanShape_;

    int stages_;
    HSize winSize_;
};
/***************************************************/

/********************** tool ***********************/
//read list file which contains many images' path
int read_file_list(const char *filePath, std::vector<std::string> &fileList);

//split file path to file directory, file name, extension
void analysis_file_path(const char* filePath, char *rootDir, char *fileName, char *ext);
void resize(uint8_t *srcData, int srcWidth, int srcHeight, int srcStride,
        uint8_t *dstData, int dstWidth, int dstHeight, int dstStride);

/***************************************************/
};
#endif
