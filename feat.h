/* Hog-like feature, implemented by Lixuan. */
/* email: 15829923@qq.com   kklotss@gmail.com */

#ifndef FEAT_H
#define FEAT_H

#include <math.h>
#define ACOS_SAMPLE_NUM 1000
#define _MIN(x,y) (((x)<(y))?(x):(y))
#define _MAX(x,y) (((x)>(y))?(x):(y))
static __inline long int
floor_f(float x)
{
    long int xi = (long int)x;
    if (x >= 0 || (float)xi == x) return xi;
    else return xi - 1;
}
struct Feat_
{
  unsigned long long numOrientations ;
  float min_cos;
  float cos_interval;
  float table_for_acos[ACOS_SAMPLE_NUM];
  /* helper vectors */
  float * orientationX ;
  float * orientationY ;

  /* buffers */
  float * feat ;
  float featNorm ;
  unsigned long long featWidth ;
  unsigned long long featHeight ;
} ;

typedef struct Feat_ Feat ;
Feat * feat_new (unsigned long long numOrientations);
void feat_delete (Feat * self) ;
void feat_put_image (Feat * self, float const * image, unsigned long long width, unsigned long long height, unsigned long long numChannels, unsigned long long cellSize) ;
void feat_extract (Feat * self, float * features) ;

#endif
