# face_alignment
A tool for face alignment

this program is a modified version of GSDM[1]. there are two major differences between this program and GSDM. First, the training samples in this program are divided into multiple subsets based on the head yaw angle, while the samples in GSDM are divided by the feature extracting from images. Second, the feature used in this program is hog-like while the feature used in GSDM is sift-like.


How to use
1. download whole program;
2. extract eigen.tar.gz and opencv2.4.11.tar.gz to ./thirdparty/ folder;
3. compile this program with visual c++ 2015 on windows;
4. run run.bat or run2.bat in ./x64/Release/ folder to see the result.


[1] Global Supervised Descent Method (https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiong_Global_Supervised_Descent_2015_CVPR_paper.pdf)
