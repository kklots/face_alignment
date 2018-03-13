/****************************************
 * Project: Joint detect
 * File: joint_face.cpp
 * Author: Hu Neng
 * Date: 2015/10/19
 *
 * Copyright 2015, Top+ vision, inc
 ****************************************/

#include "joint_face.h"

#include <float.h>
#include <list>


#define FILED_CORRECT(x, a, b) \
{ \
    if(x <= b) {\
        if(x < a) x = a;\
    } \
    else \
        x = b; \
}


#define BILINEAR_INTER(res, imgData, width, height, stride, x, y) \
{ \
    int x0 = int(x); \
    int y0 = int(y); \
    float wx = x - x0; \
    float wy = y - y0; \
    uint8_t *ptrData = imgData + y0 * stride + x0; \
    float vx0 = ptrData[0] * (1 - wx) + ptrData[1] * wx; \
    ptrData += stride; \
    float vx1 = ptrData[0] * (1 - wx) + ptrData[1] * wx; \
    res = vx0 * (1 - wy) + vx1 * wy; \
}


namespace detector{

std::vector<Point2f> MEAN_SHAPE;


Rect construct_rect(int x, int y, int width, int height){
    Rect rect;
    rect.x = x;
    rect.y = y;
    rect.width = width;
    rect.height = height;
    return rect;
}


typedef struct FeatType_t
{
    uint8_t pntIdx1, pntIdx2;
    uint8_t scale;
    float offset1_x, offset1_y;
    float offset2_x, offset2_y;
} FeatType;


typedef struct Node_t
{
    FeatType featType;
    float thresh;

    struct Node_t *left;
    struct Node_t *right;

    double score;

    int leafID;
} Node;


class CartTree
{
public:
    CartTree():depth_(0), radius_(0), ptsSize_(0), leafNum_(0){
        root_ = NULL;
        shapeOffset_ = NULL;
    }

    CartTree(int depth, int featDim, float radius, int ptsSize):
        depth_(depth), radius_(radius), ptsSize_(ptsSize), leafNum_(0){
        root_ = NULL;
        shapeOffset_ = NULL;
    }

    ~CartTree();


    int get_leaf_num(){ return leafNum_; }

    double validate(uint8_t *imgData, int width, int height, int stride, Shape &shape, float scale, float sina, float cosa, double *inc);

    void load(FILE *fin);

    double *shapeOffset_;

private:

    int depth_;
    int leafNum_;
    int ptsSize_;

    float radius_;

    Node *root_;

    uint8_t pntIdx_;
};


class BoostCart
{
public:
    BoostCart();
    ~BoostCart();

    void apply_point_offset_into_tree(std::vector<struct model*> &modXs, std::vector<struct model*> &modYs,
            int width, int height);
    int validate(uint8_t *imgData, int width, int height, int stride, Shape &shape, double &conf);

    void load(FILE *fin);

private:
    std::vector<CartTree*> carts_;
    std::vector<double> cartThresh_;

    int weakCRNum_;
    double *inc_;
};


static float diff_feature(uint8_t *img, int width, int height, int stride, Shape &shape, FeatType &featType, float s, float sina, float cosa)
{
    int x1, x2, y1, y2;
    float a, b;
    float x, y;
    float ax, ay, bx, by;

    Point2f pointA = shape[featType.pntIdx1];
    Point2f pointB = shape[featType.pntIdx2];

    sina /= s;
    cosa /= s;

    x =  (featType.offset1_x * cosa + featType.offset1_y * sina);
    y = (-featType.offset1_x * sina + featType.offset1_y * cosa);

    ax = pointA.x + x;
    ay = pointA.y + y;

    x =  (featType.offset2_x * cosa + featType.offset2_y * sina);
    y = (-featType.offset2_x * sina + featType.offset2_y * cosa);

    bx = pointB.x + x;
    by = pointB.y + y;

    FILED_CORRECT(ax, 0, (width - 1));
    FILED_CORRECT(ay, 0, (height - 2));

    FILED_CORRECT(bx, 0, (width - 1));
    FILED_CORRECT(by, 0, (height - 2));

    BILINEAR_INTER(a, img, width, height, stride, ax, ay);
    BILINEAR_INTER(b, img, width, height, stride, bx, by);

    return (a - b) + 256;
}


CartTree::~CartTree()
{
    if(root_ != NULL)
        delete [] root_;

    if(shapeOffset_ != NULL)
        delete [] shapeOffset_;
}


double CartTree::validate(uint8_t *imgData, int width, int height, int stride, Shape &shape, float scale, float sina, float cosa, double *inc)
{
    Node *node = root_;

    assert(root_ != NULL && inc != NULL);

    //depth 1
    if(node->leafID < 0){
        float value = diff_feature(imgData, width, height, stride, shape, node->featType, scale, sina, cosa);

        if(value <= node->thresh)
            node = node->left;
        else
            node = node->right;
    }

    //depth 2
    if(node->leafID < 0){
        float value = diff_feature(imgData, width, height, stride, shape, node->featType, scale, sina, cosa);

        if(value <= node->thresh)
            node = node->left;
        else
            node = node->right;
    }

    //depth 3
    if(node->leafID < 0){
        float value = diff_feature(imgData, width, height, stride, shape, node->featType, scale, sina, cosa);

        if(value <= node->thresh)
            node = node->left;
        else
            node = node->right;
    }


    assert(node != NULL && node->leafID >= 0);

    int arrSize = ptsSize_ * 2;
    double *offset = shapeOffset_ + node->leafID * arrSize;
    int len = arrSize - 4, i;

    for(i = 0; i <= len; ){
        inc[i] += offset[i]; i++;
        inc[i] += offset[i]; i++;
        inc[i] += offset[i]; i++;
        inc[i] += offset[i]; i++;
    }

    for(; i < arrSize; i++){
        inc[i] += offset[i];
    }

    return node->score;
}


void CartTree::load(FILE *fin)
{
    assert(fin != NULL);

    int ret;
    std::vector<Node*> stack;
    int nodeSize = 0;

    int ptsSize2;
    double minv, step;
    uint8_t *buf = NULL;

    ret = fread(&leafNum_, sizeof(int), 1, fin);
    ret = fread(&ptsSize_, sizeof(int), 1, fin);

    shapeOffset_ = new double[leafNum_ * ptsSize_ * 2];

    ptsSize2 = ptsSize_ * 2;
    buf = new uint8_t[ptsSize2];

    for(int i = 0; i < leafNum_; i++){
        double *ptr = shapeOffset_ + i * ptsSize2;

        ret = fread(&minv, sizeof(double), 1, fin); assert(ret == 1);
        ret = fread(&step, sizeof(double), 1, fin); assert(ret == 1);

        ret = fread(buf, sizeof(uint8_t), ptsSize2, fin); assert(ret == ptsSize2);

        for(int j = 0; j < ptsSize2; j++){
            ptr[j] = buf[j] * step + minv;
        }
    }

    delete [] buf;

    stack.push_back(new Node);
    Node *root = stack.back();

    while(stack.size() > 0){
        uint8_t flag = 0;
        Node *node = stack.back();
        stack.pop_back();
        nodeSize++;

        ret = fread(&flag, sizeof(uint8_t), 1, fin);

        if(flag == 1 || flag == 3) {
            stack.push_back(new Node);
            node->right = stack.back();
        }
        else {
            node->right = NULL;
        }

        if(flag == 2 || flag == 3) {
            stack.push_back(new Node);
            node->left = stack.back();
        }
        else {
            node->left = NULL;
        }

        if(flag != 0){
            ret = fread(&node->featType, sizeof(FeatType), 1, fin);
            ret = fread(&node->thresh, sizeof(float), 1, fin);
            node->leafID = -1;
        }
        else{
            ret = fread(&node->score, sizeof(double), 1, fin);
            ret = fread(&node->leafID, sizeof(int), 1, fin);
        }
    }

    //change to list
    root_ = new Node[nodeSize];
    int top = 1;

    memcpy(root_, root, sizeof(Node));
    delete root;

    stack.push_back(root_);

    while(stack.size() > 0){
        Node *node = stack.back();
        stack.pop_back();

        if(node->left != NULL){
            memcpy(root_ + top, node->left, sizeof(Node));
            delete node->left;
            node->left = root_ + top;
            top++;
            stack.push_back(node->left);
        }

        if(node->right != NULL){
            memcpy(root_ + top, node->right, sizeof(Node));
            delete node->right;
            node->right = root_ + top;
            top++;
            stack.push_back(node->right);
        }

        assert(top <= nodeSize);
    }
}

/****************************** Boost *****************************/

void similarity_transform(std::vector<Point2f> &shapeA, std::vector<Point2f> &shapeB, float &scale, float &angle){
    int size = shapeA.size();
    assert(size = shapeB.size());

    double per = 1.0 / size;

    double mean1x = 0.0f;
    double mean1y = 0.0f;
    double mean2x = 0.0f;
    double mean2y = 0.0f;
    double den = 0.0f, num = 0.0f, norm;
    double var1 = 0.0f, var2 = 0.0f;

    int len = size - 4, i;

    for(i = 0; i <= len; ){
        mean1x += shapeA[i].x; mean1y += shapeA[i].y; i++;
        mean1x += shapeA[i].x; mean1y += shapeA[i].y; i++;
        mean1x += shapeA[i].x; mean1y += shapeA[i].y; i++;
        mean1x += shapeA[i].x; mean1y += shapeA[i].y; i++;

        i -= 4;

        mean2x += shapeB[i].x; mean2y += shapeB[i].y; i++;
        mean2x += shapeB[i].x; mean2y += shapeB[i].y; i++;
        mean2x += shapeB[i].x; mean2y += shapeB[i].y; i++;
        mean2x += shapeB[i].x; mean2y += shapeB[i].y; i++;
    }

    for(; i < size; i++){
        mean1x += shapeA[i].x; mean1y += shapeA[i].y;
        mean2x += shapeB[i].x; mean2y += shapeB[i].y;
    }

    mean1x /= size; mean1y /= size;
    mean2x /= size; mean2y /= size;

    len = size - 4;
    for(i = 0; i <= len; ){
        var1 += sqrt(pow(shapeA[i].x - mean1x, 2) + pow(shapeA[i].y - mean1y, 2)); i++;
        var1 += sqrt(pow(shapeA[i].x - mean1x, 2) + pow(shapeA[i].y - mean1y, 2)); i++;
        var1 += sqrt(pow(shapeA[i].x - mean1x, 2) + pow(shapeA[i].y - mean1y, 2)); i++;
        var1 += sqrt(pow(shapeA[i].x - mean1x, 2) + pow(shapeA[i].y - mean1y, 2)); i++;

        i -= 4;

        var2 += sqrt(pow(shapeB[i].x - mean2x, 2) + pow(shapeB[i].y - mean2y, 2)); i++;
        var2 += sqrt(pow(shapeB[i].x - mean2x, 2) + pow(shapeB[i].y - mean2y, 2)); i++;
        var2 += sqrt(pow(shapeB[i].x - mean2x, 2) + pow(shapeB[i].y - mean2y, 2)); i++;
        var2 += sqrt(pow(shapeB[i].x - mean2x, 2) + pow(shapeB[i].y - mean2y, 2)); i++;
    }

    for(; i < size; i++){
        var1 += sqrt(pow(shapeA[i].x - mean1x, 2) + pow(shapeA[i].y - mean1y, 2));
        var2 += sqrt(pow(shapeB[i].x - mean2x, 2) + pow(shapeB[i].y - mean2y, 2));
    }

    var1 /= size;
    var2 /= size;

    scale = var1 / var2;

    for(int i = 0; i < size; i++){
        double x0 = (shapeA[i].x - mean1x) / var1;
        double y0 = (shapeA[i].y - mean1y) / var1;

        double x1 = (shapeB[i].x - mean2x) / var2;
        double y1 = (shapeB[i].y - mean2y) / var2;

        den += x0 * x1 + y0 * y1;
        num += y0 * x1 - x0 * y1;
    }

    norm = sqrt(den * den + num * num);

    angle = asin(num / norm);
}


BoostCart::BoostCart()
{
    weakCRNum_ = 0;
    inc_ = NULL;
}


BoostCart::~BoostCart()
{
    int size = carts_.size();

    for(int i = 0; i < size; i++){
        if(carts_[i] != NULL)
            delete carts_[i];
        carts_[i] = NULL;
    }

    carts_.clear();
    if(inc_ != NULL)
        delete [] inc_;

}


int BoostCart::validate(uint8_t *imgData, int width, int height, int stride, Shape &shape, double &conf)
{
    int cartSize = 0, len = 0, i, j;
    int ptsSize = shape.size();
    float scale = 0.0f, angle = 0.0f;
    float sina, cosa;

    if(inc_ == NULL)
        inc_ = new double[ptsSize * 2];

    cartSize = carts_.size();
    if(cartSize == 0)
        return 1;

    similarity_transform(MEAN_SHAPE, shape, scale, angle);

    memset(inc_, 0, sizeof(double) * ptsSize * 2);

    len = cartSize - 4;

    sina = sin(angle);
    cosa = cos(angle);

    for(i = 0; i <= len;){
        conf += carts_[i]->validate(imgData, width, height, stride, shape, scale, sina, cosa, inc_);
        if(conf <= cartThresh_[i])
            return 0;
        i++;

        conf += carts_[i]->validate(imgData, width, height, stride, shape, scale, sina, cosa, inc_);
        if(conf <= cartThresh_[i])
            return 0;
        i++;

        conf += carts_[i]->validate(imgData, width, height, stride, shape, scale, sina, cosa, inc_);
        if(conf <= cartThresh_[i])
            return 0;
        i++;

        conf += carts_[i]->validate(imgData, width, height, stride, shape, scale, sina, cosa, inc_);
        if(conf <= cartThresh_[i])
            return 0;
        i++;
    }

    for(; i < cartSize; i++){
        conf += carts_[i]->validate(imgData, width, height, stride, shape, scale, sina, cosa, inc_);
        if(conf <= cartThresh_[i])
            return 0;
    }

    sina /= scale;
    cosa /= scale;

    for(i = 0, j = 0; i < ptsSize; i++, j += 2){
        float x = inc_[j];
        float y = inc_[j + 1];

        shape[i].x +=  x * cosa + y * sina;
        shape[i].y += -x * sina + y * cosa;
    }

    return 1;
}


void BoostCart::load(FILE *fin)
{
    assert(fin != NULL);
    int ret = 0;

    ret = fread(&weakCRNum_, sizeof(int), 1, fin);

    carts_.resize(weakCRNum_);
    cartThresh_.resize(weakCRNum_);

    for(int i = 0; i < weakCRNum_; i++)
    {
        carts_[i] = new CartTree;
        carts_[i]->load(fin);

        ret = fread(&cartThresh_[i], sizeof(double), 1, fin);
    }
}


int boost_list_validate(std::vector<BoostCart*> &boosts, uint8_t *data, int width, int height, int stride, Shape &shape, double &conf){
    int size = boosts.size();
    int len = size - 5, i = 0;
    conf = 0.0f;

    for(i = 0; i <= len; i += 5){
        int ret = boosts[i]->validate(data, width, height, stride, shape, conf);
        if(ret == 0) return 0;

        ret = boosts[i + 1]->validate(data, width, height, stride, shape, conf);
        if(ret == 0) return 0;

        ret = boosts[i + 2]->validate(data, width, height, stride, shape, conf);
        if(ret == 0) return 0;

        ret = boosts[i + 3]->validate(data, width, height, stride, shape, conf);
        if(ret == 0) return 0;

        ret = boosts[i + 4]->validate(data, width, height, stride, shape, conf);
        if(ret == 0) return 0;
    }

    for(; i < size; i++){
        int ret = boosts[i]->validate(data, width, height, stride, shape, conf);
        if(ret == 0) return 0;
    }

    return 1;
}
/*******************************************************************/


/*************************** Cascade *******************************/

static void merge_rect(std::vector<Rect> &rects, std::vector<Shape> &shapes, std::vector<float> &confs)
{
    int size = rects.size();
    int *flags = NULL;
    if(size == 0)
        return;

    assert(shapes.size() == confs.size() && confs.size() == size);
    assert(size > 0);
    flags = new int[size];

    memset(flags, 0, sizeof(int) * size);

    for(int i = 0; i < size; i++)
    {
        int xi0 = rects[i].x;
        int yi0 = rects[i].y;
        int xi1 = rects[i].x + rects[i].width - 1;
        int yi1 = rects[i].y + rects[i].height - 1;

        int cix = (xi0 + xi1) / 2;
        int ciy = (yi0 + yi1) / 2;
        int sqi = rects[i].width * rects[i].height;

        for(int j = i + 1; j < size; j++)
        {
            int xj0 = rects[j].x;
            int yj0 = rects[j].y;
            int xj1 = rects[j].x + rects[j].width - 1;
            int yj1 = rects[j].y + rects[j].height - 1;

            int cjx = (xj0 + xj1) / 2;
            int cjy = (yj0 + yj1) / 2;

            int sqj = rects[j].width * rects[j].height;

            bool acInB = (xi0 <= cjx && cjx <= xi1) && (yi0 <= cjy && cjy <= yi1);
            bool bcInA = (xj0 <= cix && cix <= xj1) && (yj0 <= ciy && ciy <= yj1);
            bool acNInB = (cjx < xi0 || cjx > xi1) || (cjy < yi0 || cjy > yi1);
            bool bcNInA = (cix < xj0 || cix > xj1) || (ciy < yj0 || ciy > yj1);

            if(acInB && bcInA){
                if(confs[j] > confs[i])
                    flags[i] = 1;
                else
                    flags[j] = 1;
            }
            else if(acInB && bcNInA){
                 flags[j] = 1;
            }
            else if(acNInB && bcInA){
                flags[i] = 1;
            }
        }
    }

    for(int i = size - 1; i >= 0; i--){
        if(flags[i] == 0) continue;

        rects.erase(rects.begin() + i, rects.begin() + i + 1);
        shapes.erase(shapes.begin() + i, shapes.begin() + i + 1);
        confs.erase(confs.begin() + i, confs.begin() + i + 1);
    }

    delete []flags;
    flags = NULL;
}


Cascade::~Cascade()
{
    int size = boosts_.size();

    for(int i = 0; i < size; i++){
        if(boosts_[i] != NULL)
            delete boosts_[i];
        boosts_[i] = NULL;
    }

    boosts_.clear();
}


/*************************************************
 * startScale: min slide window / image min side
 * endScale: max side window / image min side
 *************************************************/
void Cascade::detect_patch(uint8_t *img, int width, int height, int stride, Rect &drect, int dx, int dy, float scale,
        std::vector<Rect> &rects, std::vector<Shape> &shapes, std::vector<float> &confs){

    int x0 = drect.x;
    int y0 = drect.y;
    int winW = winSize_.width;
    int winH = winSize_.height;
    int endy = drect.y + drect.height - winH;
    int endx = drect.x + drect.width - winW;

    if(winW > drect.width || winH > drect.height)
        return ;

    float scale2 = 1.15;
    int sw = drect.width * scale2;
    int sh = drect.height * scale2;
    int ss = sw;
    float scaleAll = scale * scale2;


    uint8_t *sImg = new uint8_t[sw * sh];

    resize(img + y0 * stride + x0, drect.width, drect.height, stride,
            sImg, sw, sh, ss);

    sw -= winW;
    sh -= winH;

    for(int y = 0; y < sh; y += dy){
        for(int x = 0; x < sw; x += dx){
            Shape initShape = MEAN_SHAPE;

            uint8_t *patchData = sImg + y * ss + x;
            double conf = 0;
            int ret = 0;

            ret = boost_list_validate(boosts_, patchData, winW, winH, ss, initShape, conf);

            if(ret == 1){
                int ptsSize = initShape.size();
                Rect rect;

                for(int p = 0; p < ptsSize; p++){
                    initShape[p].x = ((initShape[p].x + x) / scale2 + x0) / scale;
                    initShape[p].y = ((initShape[p].y + y) / scale2 + y0) / scale;
                }

                rect.x = (x / scale2 + x0) / scale;
                rect.y = (y / scale2 + y0) / scale;

                rect.width = winW / scaleAll;
                rect.height = winH / scaleAll;

                rects.push_back(rect);
                shapes.push_back(initShape);
                confs.push_back(conf);
            }
        }
    }

    scale2 = 0.85;
    sw = drect.width * scale2;
    sh = drect.height * scale2;
    ss = sw;
    scaleAll = scale * scale2;

    if(sw > winW && sh > winH ){
        resize(img + y0 * stride + x0, drect.width, drect.height, stride,
                sImg, sw, sh, ss);

        sw -= winW;
        sh -= winH;

        for(int y = 0; y < sh; y += dy){
            for(int x = 0; x < sw; x += dx){
                Shape initShape = MEAN_SHAPE;

                uint8_t *patchData = sImg + y * ss + x;
                double conf = 0;
                int ret = 0;

                ret = boost_list_validate(boosts_, patchData, winW, winH, ss, initShape, conf);

                if(ret == 1){
                    int ptsSize = initShape.size();
                    Rect rect;

                    for(int p = 0; p < ptsSize; p++){
                        initShape[p].x = ((initShape[p].x + x) / scale2 + x0) / scale;
                        initShape[p].y = ((initShape[p].y + y) / scale2 + y0) / scale;
                    }

                    rect.x = (x / scale2 + x0) / scale;
                    rect.y = (y / scale2 + y0) / scale;

                    rect.width = winW / scaleAll;
                    rect.height = winH / scaleAll;

                    rects.push_back(rect);
                    shapes.push_back(initShape);
                    confs.push_back(conf);
                }
            }
        }
    }


    /*
    for(int y = y0; y < endy; y += dy){
        for(int x = x0; x < endx; x += dx){
            Shape initShape = MEAN_SHAPE;

            uint8_t *patchData = img + y * stride + x;
            double conf = 0;
            int ret = 0;

            ret = boost_list_validate(boosts_, patchData, winW, winH, stride, initShape, conf);

            if(ret == 1){
                int ptsSize = initShape.size();
                Rect rect;

                for(int p = 0; p < ptsSize; p++){
                    initShape[p].x = (initShape[p].x + x) / scale;
                    initShape[p].y = (initShape[p].y + y) / scale;
                }

                rect.x = x / scale;
                rect.y = y / scale;
                rect.width = winSize_.width / scale;
                rect.height = winSize_.height / scale;

                rects.push_back(rect);
                shapes.push_back(initShape);

                confs.push_back(conf);
            }
        }
    }
    */

    delete [] sImg;
}


void Cascade::detect(uint8_t *img, int width, int height, int stride, std::vector<Rect> &rects, std::vector<Shape > &shapes,
        float startScale, float endScale, int layers, float offsetFactor)
{
    std::vector<float> confs;
    detect(img, width, height, stride, rects, shapes, confs, startScale, endScale, layers, offsetFactor);
}

void Cascade::detect(uint8_t *img, int width, int height, int stride, std::vector<Rect> &rects, std::vector<Shape > &shapes,
        std::vector<float> &confs, float startScale, float endScale, int layers, float offsetFactor){
    uint8_t *sImg;
    uint8_t *buffer;

    int sw, sh, ss;
    int bWidth, bHeight, bStride;

    float scale;
    int dx, dy;
    int minSide = HU_MIN(width, height);

    float stepScale = 0.0;
    int winW = winSize_.width;
    int winH = winSize_.height;

    if(img == NULL) return ;

    if(startScale < 0.1f || startScale > 0.5f || endScale < 0.5f || endScale > 1.0f || endScale < startScale ){
        startScale = 0.2f;
        endScale = 0.8f;
        offsetFactor = 0.2f;
    }

    if(layers == 0) layers = 5;

    if(offsetFactor < 0.05f)
        offsetFactor = 0.05;
    else if(offsetFactor >= 0.6f)
        offsetFactor = 0.6;

    stepScale = (endScale - startScale) / layers;

    scale = HU_MAX(winH, winW) / (minSide * startScale);

    sw = width * scale;
    sh = height * scale;
    sImg = new uint8_t[sw * sh];

    bStride = HU_MAX(sw, stride);
    bHeight = HU_MAX(sh, height);
    buffer = new uint8_t[bStride * bHeight];
    memcpy(buffer, img, stride * height * sizeof(uint8_t));

    bWidth = width;
    bHeight = height;
    bStride = stride;

    dx = winW * offsetFactor;
    dy = winH * offsetFactor;

    rects.clear();
    shapes.clear();
    confs.clear();


    for(int i = 0; i < layers; i++){
        std::vector<Rect> lrects;
        std::vector<Shape> lshapes;
        std::vector<float> lconfs;

        int w, h;

        float side = minSide * startScale;
        scale = HU_MAX(winW, winH) / side;

        sw = scale * width;
        sh = scale * height;
        ss = sw;

        resize(buffer, bWidth, bHeight, bStride, sImg, sw, sh, ss);

        w = sw - winW;
        h = sh - winH;

        int startK = rects.size();
        for(int y = 0; y < h; y += dy){
            for(int x = 0; x < w; x+= dx){
                Shape initShape = meanShape_;

                uint8_t *patchData = sImg + y * ss + x;
                double conf = 0;
                int ret = 0;

                ret = boost_list_validate(boosts_, patchData, winW, winH, ss, initShape, conf);

                if(ret == 1){
                    int ptsSize = initShape.size();
                    Rect rect;

                    rect.x = x;
                    rect.y = y;
                    rect.width = winW;
                    rect.height = winH;

                    lrects.push_back(rect);
                    lshapes.push_back(initShape);
                    lconfs.push_back(conf);

                    rect.x /= scale;
                    rect.y /= scale;
                    rect.width /= scale;
                    rect.height /= scale;

                    for(int i = 0; i < ptsSize; i++){
                        initShape[i].x = (initShape[i].x + x) / scale;
                        initShape[i].y = (initShape[i].y + y) / scale;
                    }

                    rects.push_back(rect);
                    shapes.push_back(initShape);
                    confs.push_back(conf);
                }
            }
        }


        merge_rect(lrects, lshapes, lconfs);

        int lsize = lrects.size();

        for(int j = 0; j < lsize; j++){
            Rect rect = lrects[j];

            rect.x -= 0.3 * rect.width;
            rect.y -= 0.3 * rect.height;

            rect.width *= 1.6;
            rect.height *= 1.6;

            rect.x = HU_MAX(0, rect.x);
            rect.y = HU_MAX(0, rect.y);

            rect.width = HU_MIN(rect.width, sw - rect.x);
            rect.height = HU_MIN(rect.height, sh - rect.y);

            detect_patch(sImg, sw, sh, ss, rect, 8, 8, scale, rects, shapes, confs);
        }

        startScale += stepScale;

        {
            uint8_t *t = buffer; buffer = sImg; sImg = t;
            bWidth = sw, bHeight = sh, bStride = ss;
        }
    }

    merge_rect(rects, shapes, confs);

    delete [] buffer;
    delete [] sImg;
}


int Cascade::validate(uint8_t *img, int width, int height, int stride){
    Shape shape = MEAN_SHAPE;
    double conf = 0.0;
    uint8_t *sImg = NULL;

    int flag = (width == winSize_.width && height == winSize_.height);
    int stride2, ret;

    if(flag){
        sImg = img;
        stride2 = stride;
    }
    else {
        sImg = new uint8_t[winSize_.width * winSize_.height];
        resize(img, width, height, stride, sImg, winSize_.width, winSize_.height, winSize_.width);

        stride2 = winSize_.width;
    }

    int bSize = boosts_.size();

    for(int i = 0; i < bSize; i++){
        ret = boosts_[i]->validate(sImg, winSize_.width, winSize_.height, stride2, shape, conf);
        if(ret == 0) break;
    }

    if(!flag) delete [] sImg;


    return ret;
}


int Cascade::load(const char *fileName)
{
    int ret, ptsSize;
    FILE *fin = fopen(fileName, "rb");

    if(fin == NULL)
    {
        printf("Can't open file %s\n", fileName);
        return 1;
    }

    ret = fread(&winSize_, sizeof(HSize), 1, fin);
    ret = fread(&stages_, sizeof(int), 1, fin);

    boosts_.resize(stages_);

    for(int i = 0; i < stages_; i++){
        boosts_[i] = new BoostCart;
        boosts_[i]->load(fin);
    }


    ret = fread(&ptsSize, sizeof(int), 1, fin);
    meanShape_.resize(ptsSize);

    for(int i = 0; i < ptsSize; i++){
        ret = fread(&meanShape_[i], sizeof(Point2f), 1, fin);
    }

    fclose(fin);

    MEAN_SHAPE = meanShape_;
    return 0;
}
/*******************************************************************/


/************************** Utils **********************************/
int read_file_list(const char *filePath, std::vector<std::string> &fileList)
{
    char line[512];
    FILE *fin = fopen(filePath, "r");

    if(fin == NULL){
        printf("Can't open file: %s\n", filePath);
        return -1;
    }

    while(fscanf(fin, "%s\n", line) != EOF){
        fileList.push_back(line);
    }

    fclose(fin);

    return 0;
}


void analysis_file_path(const char* filePath, char *rootDir, char *fileName, char *ext)
{
    int len = strlen(filePath);
    int idx = len - 1, idx2 = 0;

    while(idx >= 0){
        if(filePath[idx] == '.')
            break;
        idx--;
    }

    if(idx >= 0){
        strcpy(ext, filePath + idx + 1);
        ext[len - idx] = '\0';
    }
    else {
        ext[0] = '\0';
        idx = len - 1;
    }

    idx2 = idx;
    while(idx2 >= 0){
#ifdef WIN32
        if(filePath[idx2] == '\\')
#else
        if(filePath[idx2] == '/')
#endif
            break;
        idx2 --;
    }

    if(idx2 > 0){
        strncpy(rootDir, filePath, idx2);
        rootDir[idx2] = '\0';
    }
    else{
        rootDir[0] = '.';
        rootDir[1] = '\0';
    }

    strncpy(fileName, filePath + idx2 + 1, idx - idx2 - 1);
    fileName[idx-idx2-1] = '\0';
}


typedef struct {
    float w0, w1;
    int idx;
} AlphaInfo;


AlphaInfo * create_table(int src, int dst){
    AlphaInfo *table = new AlphaInfo[dst];

    float scale = float(src) / dst;
    float idx = 0.0;

    int len = dst - 4;

    for(int i = 0; i <= len; ){
        idx = i * scale;

        table[i].idx = int(idx);
        table[i].w0 = idx - table[i].idx;
        table[i].w1 = 1 - table[i].w0;
        i++;

        idx = i * scale;

        table[i].idx = int(idx);
        table[i].w0 = idx - table[i].idx;
        table[i].w1 = 1 - table[i].w0;
        i++;

        idx = i * scale;

        table[i].idx = int(idx);
        table[i].w0 = idx - table[i].idx;
        table[i].w1 = 1 - table[i].w0;
        i++;

        idx = i * scale;

        table[i].idx = int(idx);
        table[i].w0 = idx - table[i].idx;
        table[i].w1 = 1 - table[i].w0;
        i++;
    }

    for(int i = len + 1; i < dst; i++){
         idx = i * scale;

        table[i].idx = int(idx);
        table[i].w0 = idx - table[i].idx;
        table[i].w1 = 1 - table[i].w0;
    }

    return table;
}


void resize(uint8_t *srcData, int srcWidth, int srcHeight, int srcStride,
        uint8_t *dstData, int dstWidth, int dstHeight, int dstStride)
{
    float wy0,  wy1;
    float value1, value2;
    uint8_t *ptrData = NULL, *ptrData2 = NULL;
    AlphaInfo *colt = create_table(srcWidth, dstWidth);
    AlphaInfo *rowt = create_table(srcHeight, dstHeight);

    colt[dstWidth - 1].idx -= 1;
    rowt[dstHeight - 1].idx -= 1;

    assert(srcData != NULL && dstData != NULL);
    assert(srcWidth > 0 && dstWidth > 0 && srcHeight > 0 && dstHeight > 0);

    for(int y = 0; y < dstHeight; y++){
        int idx, idx2;
        int len = dstWidth - 4;
        wy0 = rowt[y].w0;
        wy1 = rowt[y].w1;
        idx = rowt[y].idx * srcStride;

        for(int x = 0; x <= len; ){
            idx2 = idx + colt[x].idx;

            value1 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1]; idx2 += srcStride;
            value2 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1];

            dstData[x] = wy1 * value1 + wy0 * value2; x++;

            idx2 = idx + colt[x].idx;

            value1 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1]; idx2 += srcStride;
            value2 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1];

            dstData[x] = wy1 * value1 + wy0 * value2; x++;

            idx2 = idx + colt[x].idx;

            value1 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1]; idx2 += srcStride;
            value2 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1];

            dstData[x] = wy1 * value1 + wy0 * value2; x++;

            idx2 = idx + colt[x].idx;

            value1 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1]; idx2 += srcStride;
            value2 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1];

            dstData[x] = wy1 * value1 + wy0 * value2; x++;
        }

        for(int x = len + 1; x < dstWidth; x++){
            idx2 = idx + colt[x].idx;

            value1 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1]; idx2 += srcStride;
            value2 = colt[x].w1 * srcData[idx2] + colt[x].w0 * srcData[idx2 + 1];

            dstData[x] = wy1 * value1 + wy0 * value2;
        }

        dstData += dstStride;
    }

    delete [] colt;
    delete [] rowt;
}


/*******************************************************************/

};
