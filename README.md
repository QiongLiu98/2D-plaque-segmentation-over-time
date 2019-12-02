# 2D-plaque-segmentation-over-time
U-Net plaque segmentation
2D plaque segmentation at different time-points
Please contact me if you have any questions:
liuqiong1998@hust.edu.cn
qiongliu98@gmail.com


1.Extract images to /home/ran/2DPlaqueData_SPARC


Matlab
2.Generate csv file (contains image path and mask path) at /home/ran/MyProjs/Get2DSegData/11-getcsv_file_chris.m 


Pycharm (active gpu before you run the code by: conda activate tensorflow-gpu)
csv file 
csv file contains plaques with different time-points is for testing;
copy and past it to /home/ran/MyProjs/2DPlaqueSeg/Datalist_2DPlaque_SPARCM_timepoint_1128.csv

3.Load data from csv file into npy data at
/home/ran/MyProjs/2DPlaqueSeg/graham_SPARC_fixedS/Data_Augmentation2D.py
549 csv file
561 npy data folder
443 number of testing data

4.Train U-Net by using npydata and save the model weights at the log folder
/home/ran/MyProjs/2DPlaqueSeg/graham_SPARC_fixedS/11-Train_plaque_unet.py 
15 load npy data path
19 save model weights path

5.Generate U-Net predicted masks by loading weights from log folder at
/home/ran/MyProjs/2DPlaqueSeg/graham_SPARC_fixedS/11-test_predict_plaque_augment_original.py 
20 load csv file path (with plaques at different time-points)
21 generated images folder
22 U-Net model weights save path


Matlab
6.Sort out predicted images from mixed to (patient_number/timepoints/plaque position/plaque name) at
/home/ran/MyProjs/2DPlaqueSeg/results/sort_out_images.m
12 mixed images path
11 saved root path
14 original image path
7.Calculate area and parameters and save them in a csv file at
/home/ran/MyProjs/2DPlaqueSeg/results/calculate_total_plaque_area.m
9 define sort image folder
15&16 Define csv_file_name
