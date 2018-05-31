# face_alignment

2018.5.31: updated the trainning scripts.

a implement of face alignment.

this program is a modified version of GSDM[1]. there are two major differences between this program and GSDM. First, the training samples in this program are divided into multiple subsets based on the head yaw angle, while the samples in GSDM are divided by the feature extracting from images. Second, the feature used in this program is hog-like while the feature used in GSDM is sift-like.


How to use
1. download whole program;
2. extract eigen.tar.gz and opencv2.4.11.tar.gz to ./thirdparty/  ;
3. compile this program with visual c++ 2012_x64_release on windows;
4. run run.bat or run2.bat in ./x64/Release/ to see the result. (the models are in folder 'x64/Release/models/')


[1] Global Supervised Descent Method (https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiong_Global_Supervised_Descent_2015_CVPR_paper.pdf)
