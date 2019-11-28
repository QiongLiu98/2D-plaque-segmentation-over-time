%two manual segmentation comparision test
clc
clear all
close all

folder_M = '/home/ran/MyProjs/Get2DSegData/2DplaqueM_timepoints_qiongtest/masks-M';
folder_C = '/home/ran/2DPlaqueData_SPARC/PLAQUE_comparision_test';

objects_M = dir(folder_M);
objects_C = dir(folder_C);

csv_file = './report_roi_qiong/plaque_area_comparision1125.csv';

plaque_fid_each = fopen(csv_file, 'a');
fprintf(plaque_fid_each,' %s, %s, %s, %s, %s, %s, %s, %s, %s , %s, %s, %s \r\n',...
    'image_name', ...
    'area_m', 'area_c', 'area_arror(m-c)', ...
    'dice', 'sen', 'sp', 'acc',...
    'width_manual', 'width_pred', 'height_manual', 'height_pred');
avg_dice  = 0;
 for i = 3:length(objects_M)
     imagename_M = objects_M(i).name;
     imagename_C = objects_C(i).name;
     imagepath_M = fullfile(folder_M, imagename_M);
     imagepath_C = fullfile(folder_C, imagename_C);
     image_M = imread(imagepath_M);
     image_C = logical(rgb2gray(imread(imagepath_C)));
        %get parameters

        %area arror

        bi_m = im2bw(image_M);
        bi_c = im2bw(image_C);
        area_M = length(find(bi_m==1));
        area_C = length(find(bi_c==1));
        area_m_c = area_M-area_C;

    

        %DICE
        area_and = length(find(bi_m & bi_c==1));
        area_or = length(find(bi_m | bi_c==1));
        dice = 2*area_and/(area_C+area_M);
        %sen sp acc
        [a,b] = size(bi_c);
        tp = area_and;
        tn = a*b-area_or;
        fp = area_C-area_and;
        fn = area_M-area_and;
        sen = tp/(tp+fn);
        sp = tn/(fp+tn);
        acc = (tp+tn)/(tp+tn+fp+fn);
        %generate a single report for this certain plaque

        %width and height
        [h_m,w_m] = find(bi_m==1);
        width_m = max(w_m)-min(w_m);
        height_m = max(h_m)-min(h_m);
        [h_c,w_c] = find(bi_c==1);
        width_c = max(w_c)-min(w_c);
        height_c = max(h_c)-min(h_c);
        fprintf(plaque_fid_each,' %s,  %d, %d, %d, %d, %d, %d, %d,%d, %d, %d, %d\r\n', ...
        imagename_M, ...
        area_M*0.065*0.065, area_C*0.065*0.065,  area_m_c*0.065*0.065, ...
        dice, sen, sp, acc,...
        width_m, width_c, height_m, height_c);
 avg_dice = dice + avg_dice;
 end
 avg_dice = avg_dice/63