%This code generates a csv file which contains
%'patient_name','position','time','plaque_name', ...
%'area_man', 'area_pred', 'area_avg', 'area_pred_man', 'area_man_pred', ...
%'dice', 'sen', 'sp', 'acc'
clc
clear all
close all

sort_image_folder = './generate_report_chris_1120';

%generate progression report for each plaque
report_root = dir(sort_image_folder);

%create a csv file contains all information
csv_file_name_each = './report_roi_qiong/plaque_area1120.csv';
csv_file_name_total = './report_roi_qiong/plaque_area_total1120.csv';

plaque_fid_each = fopen(csv_file_name_each, 'a');
fprintf(plaque_fid_each,' %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s , %s, %s, %s, %s \r\n',...
    'patient_name', 'position_name', 'time_name', 'plaque_name', 'number', ...
    'area_man', 'area_pred', 'area_avg', 'area_arror(pred-man)', ...
    'dice', 'sen', 'sp', 'acc',...
    'width_manual', 'width_pred', 'height_manual', 'height_pred');
% fprintf(plaque_fid_each,' %s, %s, %s, %s, %s\r\n',...
%     'patient_name','position','time','plaque_name','plaque_part', ...
%      'area_pred');
plaque_fid_total = fopen(csv_file_name_total, 'a');
fprintf(plaque_fid_total,' %s, %s, %s, %s\r\n',...
    'patient_name','time', 'area_manual','area_pred');
for i_root = 3:length(report_root)%patient name
    %i_root = 3;
    patient_name = report_root(i_root).name;
    report_patient_path = fullfile(sort_image_folder,report_root(i_root).name);
    report_patient_name = dir(report_patient_path);
    
    for i_time = 3:length(report_patient_name)%time
        %i_time = 3
        time_name = report_patient_name(i_time).name;
        report_time_path = fullfile(report_patient_path,time_name);
        report_time_name = dir(report_time_path);
        %change time_name to generate chart easier
        time_seg = strfind(time_name,'-');
        time_year = time_name(time_seg(2)+1:end);
        time_month_day = time_name(1:time_seg(2)-1);
        time_fixed = [time_year,'-',time_month_day];
        %creat a csv file
        area_pred_total = 0;
        area_man_total = 0;
        
        for i_position = 3:length(report_time_name)
            %i_position = 3
            position_name = report_time_name(i_position).name;
            report_position_path = fullfile(report_time_path,position_name);
            report_position_name = dir(report_position_path);
           
            for i_plaque = 3:length(find([report_position_name.isdir]==1))
                %i_plaque = 3
                plaque_name = report_position_name(i_plaque).name;
                report_plaque_path = fullfile(report_position_path,plaque_name);
                report_plaque_name = dir(report_plaque_path);
                
                
                for i_number = 3:length(report_plaque_name)
                    if rem(i_number-3,3)==1
                        image_pred_name = report_plaque_name(i_number+1).name;
                        image_pred = imread(fullfile(report_plaque_path,image_pred_name));

                        image_man_name = report_plaque_name(i_number).name;
                        image_man = imread(fullfile(report_plaque_path,image_man_name));
                        
                        number_locs = strfind(image_pred_name,'(');
                        number = image_pred_name(number_locs+1);
                       

                        %get parameters

                        %area arror

                        bi_man = im2bw(image_man);
                        bi_pred = im2bw(image_pred);
                        area_man = length(find(bi_man==1));
                        area_pred = length(find(bi_pred==1));
                        area_avg = (area_man+area_pred)/2;
                        area_pred_man = area_pred-area_man;

                        area_pred_total = area_pred + area_pred_total;
                        area_man_total = area_man + area_man_total;

                        %DICE
                        area_and = length(find(bi_man & bi_pred==1));
                        area_or = length(find(bi_man | bi_pred==1));
                        dice = 2*area_and/(area_pred+area_man);
                        %sen sp acc
                        [a,b] = size(bi_pred);
                        tp = area_and;
                        tn = a*b-area_or;
                        fp = area_pred-area_and;
                        fn = area_man-area_and;
                        sen = tp/(tp+fn);
                        sp = tn/(fp+tn);
                        acc = (tp+tn)/(tp+tn+fp+fn);
                        %generate a single report for this certain plaque
                        
                        %width and height
                        [h_m,w_m] = find(bi_man==1);
                        width_manual = max(w_m)-min(w_m);
                        height_manual = max(h_m)-min(h_m);
                        [h_p,w_p] = find(bi_pred==1);
                        width_pred = max(w_p)-min(w_p);
                        height_pred = max(h_p)-min(h_p);
                fprintf(plaque_fid_each,' %s, %s, %s, %s, %s, %d, %d, %d, %d, %d, %d, %d, %d,%d, %d, %d, %d\r\n', ...
                    patient_name, position_name, time_name, plaque_name, number, ...
                    area_man*0.065*0.065, area_pred*0.065*0.065, area_avg*0.065*0.065, area_pred_man*0.065*0.065, ...
                    dice, sen, sp, acc,...
                    width_manual, width_pred, height_manual, height_pred);
                    end
                end
                    
               

                
%                         fprintf(plaque_fid_each,' %s, %s, %s, %s, %d, %d\r\n', ...
%                                             patient_name, position_name, time_name, plaque_name, ...
%                                             i_part, area_pred);
            end

                end
               
             
            
        
        %area_man_total = area_man_total*0.065*0.065;
        
%         fprintf(plaque_fid_total,' %s, %s, %d, %d, %d \r\n',...
%         patient_name,time_fixed, ...
%         area_man_total, area_pred_total, area_pred_total-area_man_total);
         fprintf(plaque_fid_total,' %s, %s, %d, %d \r\n',...
                patient_name,time_fixed, ...
                area_man_total*0.065*0.065,area_pred_total*0.065*0.065);
        
   
   end
       
        end 


