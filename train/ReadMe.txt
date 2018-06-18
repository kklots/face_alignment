How to use the training scripts:

1. very limited samples are given in data0, data1 and data2 to define the label format and the sign of the yaw angle.

2. prepare labeled training samples as many as you can, convert them to gray and resize the face area to 200*200 ( the margin length is 100, so the final size of the samples is 400*400 );

3. devide the samples into 5 folders according to yaw angle. folder_1: -90~-54, folder_2: -54~-18, folder_3: -18~18, folder_4: 18~54, folder_5: 54~90;

4. copy the samples in folder_2, folder_3, and folder_4 to './data0/', the samples in folder_1, folder_2, and folder_3 to './data1/', and the samples in folder_3, folder_4, and folder_5 to './data2/';

5. run trainScript.m to train model.

6. run aSimpleExample.m to get a rough model. the trainning samples for aSimpleExample.m script can be downloaded from https://pan.baidu.com/s/1y4Wt_OJGodA3G4iQujQnJg ( password: 1d49 ).

enjoy yourself.  :)
