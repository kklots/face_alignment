/* Hog-like feature, implemented by Lixuan. */
/* email: 15829923@qq.com   kklotss@gmail.com */

#include "feat.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <Eigen/Dense>

Feat *
feat_new (unsigned long long numOrientations)
{
  long long o, k ;
  Feat * self =(Feat *) calloc(1, sizeof(Feat)) ;

  assert(numOrientations >= 1) ;
  self->numOrientations = numOrientations ;
  self->orientationX = (float*) malloc(sizeof(float) * self->numOrientations) ;
  self->orientationY = (float*) malloc(sizeof(float) * self->numOrientations) ;
  self->min_cos = 0;
  self->cos_interval = 1.0f/ACOS_SAMPLE_NUM;
  for(int i=0;i<ACOS_SAMPLE_NUM;i++)
  {
	  self->table_for_acos[i] = std::fmod(acosf(self->min_cos+ self->cos_interval*i),3.141592653589793/self->numOrientations) *self->numOrientations * 0.3183098861837906956990848088196;
  }

  for(o = 0 ; o < (signed)self->numOrientations ; ++o) {
    float angle = o * 3.14159265358979323846 / self->numOrientations ;
    self->orientationX[o] = (float) cos(angle) ;
    self->orientationY[o] = (float) sin(angle) ;

  }
  return self ;
}

void
feat_delete (Feat * self)
{
  if (self->orientationX) {
    free(self->orientationX) ;
    self->orientationX = NULL ;
  }

  if (self->orientationY) {
    free(self->orientationY) ;
    self->orientationY = NULL ;
  }
  if (self->feat) {
    free(self->feat) ;
    self->feat = NULL ;
  }
  free(self) ;
}

static void
feat_prepare_buffers (Feat * self, unsigned long long width, unsigned long long height, unsigned long long cellSize)
{
  unsigned long long featWidth = (width + cellSize/2) / cellSize ;
  unsigned long long featHeight = (height + cellSize/2) / cellSize ;
  assert(width > 3) ;
  assert(height > 3) ;
  assert(featWidth > 0) ;
  assert(featHeight > 0) ;

  if (self->feat &&
      self->featWidth == featWidth &&
      self->featHeight == featHeight) {
    /* a suitable buffer is already allocated */
    memset(self->feat, 0, sizeof(float) * featWidth * featHeight * self->numOrientations * 2) ;
	self->featNorm = 0;
    return ;
  }

  if (self->feat) {
    free(self->feat) ;
    self->feat = NULL ;
  }
  self->feat = (float*) calloc(featWidth * featHeight * self->numOrientations * 2, sizeof(float)) ;
  self->featNorm = 0;
  self->featWidth = featWidth ;
  self->featHeight = featHeight ;
}

void
feat_put_image (Feat * self,
                  float const * image,
                  unsigned long long width, unsigned long long height, unsigned long long numChannels,
                  unsigned long long cellSize)
{
  unsigned long long featStride ;
  unsigned long long channelStride = width * height ;
  long long x, y ;
   unsigned long long k ;

  assert(self) ;
  assert(image) ;

  /* clear features */
  feat_prepare_buffers(self, width, height, cellSize) ;
  
  featStride = self->featWidth * self->featHeight ;

#define at(x,y,k) (self->feat[(x) + (y) * self->featWidth + (k) * featStride])

  float hx, hy, wx1, wx2, wy1, wy2 ; 
  long long binx, biny, o ;
  
  /* compute gradients and map the to feat cells by bilinear interpolation */
  for (y = 1 ; y < (signed)height - 1 ; y+=1) {
	hy = (0.5+y) / float(cellSize) -0.5;
	biny = floor_f(hy) ;	
	wy2 = hy - biny ;	
	wy1 = 1.0 - wy2 ;
    for (x = 1 ; x < (signed)width - 1 ;  x+=1) {
      float gradx = 0 ;
      float grady = 0 ;
      float gradNorm ;
      float orientationWeights [2] = {-1, -1} ;
      long long orientationBins [2] = {-1, -1} ;
      long long orientation = 0 ;
      {
        float const * iter = image + y * width + x ;
        float gradNorm2 = 0 ;
        for (k = 0 ; k < numChannels ; ++k) {
          float gradx_ = *(iter + 1) - *(iter - 1) ;
          float grady_ = *(iter + width)  - *(iter - width) ;
          float gradNorm2_ = gradx_ * gradx_ + grady_ * grady_ ;
          if (gradNorm2_ > gradNorm2) {
            gradx = gradx_ ;
            grady = grady_ ;
            gradNorm2 = gradNorm2_ ;
          }
          iter += channelStride ;
        }
		gradNorm = gradNorm2> 1e-20 ? sqrtf(gradNorm2):1e-10;
      }

      for (k = 0 ; k < self->numOrientations ; ++k) {
        float orientationScore_ = (gradx * self->orientationX[k] +  grady * self->orientationY[k])/gradNorm;
        long long orientationBin_ = k ;
        if (orientationScore_ < 0)
		 {
          orientationScore_ = - orientationScore_ ;
          orientationBin_ += self->numOrientations ;
        }
		
        if (orientationScore_ > orientationWeights[0]) {
          orientationBins[1] = orientationBins[0] ;
          orientationWeights[1] = orientationWeights[0] ;
          orientationBins[0] = orientationBin_ ; ;
          orientationWeights[0] = orientationScore_ ;
        } else if (orientationScore_ > orientationWeights[1]) {
          orientationBins[1] = orientationBin_ ;
          orientationWeights[1] = orientationScore_ ;
        }
      }
	  float cos_score = _MIN(orientationWeights[0] , 1.0);
	  if(cos_score ==1)
	  {
		  orientationWeights[0] = 1 ;
		  orientationBins[1] = -1 ;
	  }else
	  {
		  int acos_idx = ( cos_score - self->min_cos ) / self->cos_interval;
		  float param1 = self->table_for_acos[acos_idx];
		  orientationWeights[1] = param1<1-param1?param1:1-param1;
		  orientationWeights[0] = 1 - orientationWeights[1] ;
	  }
	  
	  hx = (0.5+x) / float(cellSize) -0.5;
	  binx = floor_f(hx) ;
	  wx2 = hx - binx ;
	  wx1 = 1.0 - wx2 ;

      for (o = 0 ; o < 2 ; ++o) {
        float ow ;
        orientation = orientationBins[o] ;
        if (orientation < 0) continue ;

        ow = orientationWeights[o] ;

        if (binx >= 0 && biny >=0) {
          at(binx,biny,orientation) += gradNorm * ow * wx1 * wy1 ;
        }
        if (binx < (signed)self->featWidth - 1 && biny >=0) {
          at(binx+1,biny,orientation) += gradNorm * ow * wx2 * wy1 ;
        }
        if (binx < (signed)self->featWidth - 1 && biny < (signed)self->featHeight - 1) {
          at(binx+1,biny+1,orientation) += gradNorm * ow * wx2 * wy2 ;
        }
        if (binx >= 0 && biny < (signed)self->featHeight - 1) {
          at(binx,biny+1,orientation) += gradNorm * ow * wx1 * wy2 ;
        }
      } /* next o */
    } /* next x */
  } /* next y */

  {
	float * iter = self->feat; 
	unsigned long long stride = self->featWidth*self->featHeight*self->numOrientations ;
    float * iterEnd = self->feat  + stride;

    for (k = 0 ; k < self->numOrientations ; ++k) {
      while (iter != iterEnd) {
        float h1 = *iter ;
        float h2 = *(iter + stride) ;
        float h = h1 + h2 ;
        self->featNorm += h * h ;
        iter++ ;
      }
    }
  }
}

void
feat_extract (Feat * self, float * features)
{
  long x, y ;
  unsigned long long k ;
  unsigned long long  featStride = self->featWidth * self->featHeight ;
  float denominator =  float((self->featWidth) * (self->featHeight));
  assert(features) ;
  
  float factor ;
  factor = 1.0/(denominator*sqrtf(self->featNorm+1e-9));

  { 
    for (y = 0 ; y < self->featWidth; y+=1) {
      for (x = 0 ; x < self->featWidth; x+=1) {

        float text = 0 ;
        float * oiter = features + x + self->featWidth * y;

        for (k = 0 ; k < self->numOrientations ; ++k)
		{
			float ha=0;
			float hb=0;
			ha+=at(x,y,k);
			hb+=at(x,y,k + self->numOrientations);
    		ha *= factor;
			hb *= factor;
			float hc = ha + hb;
			ha = _MIN(0.2, ha) ;
			hb = _MIN(0.2, hb) ;
			hc = _MIN(0.2, hc) ;
			text += hc ;
			*oiter = ha;
            *(oiter + featStride * self->numOrientations) = hb ;
            *(oiter + 2 * featStride * self->numOrientations) = hc ;
			oiter += featStride ;

		} /* next orientation */
        oiter += 2 * featStride * self->numOrientations ;
        *oiter = 0.2357 * text ;
      } /* next x */
    } /* next y */
  } /* block normalization */
}