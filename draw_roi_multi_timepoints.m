clc
clear all;
close all;
path_image = '/home/ran/2DPlaqueData_SPARC/US-carotid-images-multiple-timepoints/11';

savepath = './1120_roi/2DplaqueQ';
savepath_roi = './1120_roi/2DplaqueQ_with_ROI';
csvout = './1120_roi/Datalist_qiong_1113.csv';
%csv_info = './1120_roi/plaque_info.csv';

if exist(savepath,'dir')==0
    mkdir(savepath);
end
if exist(savepath_roi,'dir')==0
    mkdir(savepath_roi);
end
objects = dir(path_image);

fid = fopen(csvout, 'a');
fprintf(fid,'%s, %s, %s, %s, %s, %s\r\n','filepath', 'minx','maxx','miny','maxy','res');
% fid2 = fopen(csv_info,'a');
% fprintf(fid2,'%s, %s, %s, %s\r\n','patient','time','doc_area','missing_num');
%mth = {'Jan', 'Feb', 'Mar', 'Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};


for idx_obj = 8:length(objects)
    %idx_obj = 3;
    patient_name = objects(idx_obj).name;
    patient_folder_name = fullfile(path_image, patient_name);
    patient_folder = dir(patient_folder_name);%./patient number

    for idx_time = 3 : length(patient_folder)
        %idx_time = 3;
        time_name = patient_folder(idx_time).name;% get one time point
        time_folder_name = fullfile(patient_folder_name, time_name)
        time_folder = dir(time_folder_name);%./time
%         if exist(time_folder_name, 'dir') 
%             tmplocs = strfind(time_name, '-');
%             substr = time_name(1 : tmplocs(1) - 1);
%             for tmpidx = 1 : length(mth)% function diceco = Dice(BW1, BW2)
%                 if strcmp(substr, mth{tmpidx})
%                     break;
%                 end
%             end
%             
%             time_name_new = sprintf('%02d%s', tmpidx, time_name(tmplocs(1): end)); 
% 
%             time_folder_name_new = fullfile(patient_folder_name, time_name_new);%change Feb to 02
%         end
        
%         figure(1)
%         report_path = fullfile(time_folder_name,time_folder(length(time_folder)).name);
%         report = imread(report_path);
%         re = imshow(report);
        
%         doc_area = input('doc_area:');
%         missing_plaques_number = input('missing plaque number:');
%         fprintf(fid2,'%s, %s, %d, %d\r\n', patient_name, time_name, doc_area, missing_plaques_number);
        for idx_images = 3 : length(time_folder)
            %idx_images = 3
            imagename = time_folder(idx_images).name;
            imagepath = fullfile(time_folder_name,imagename);
            imglocs1 = strfind(imagename,'-');
            imglocs2 = strfind(imagename,'.');
            i=0;
            image = imread(imagepath);
            im = imshow(image);
            title(imagepath)
            
            resulution = input('resolution(input 0 if no plaque):');
            if resulution == 0
                continue;
            end
            
            while(1)
                i = i+1;
                mouse=imrect;
               
                position = input ('position(eg:RTICA):','s');
                plaque_position = input('l/r/m):','s');
                plaque_name = input('plaque_name(eg:1/2/3):');
                
                
                pos=getPosition(mouse);% x1 y1 w h
                minx = pos(1); miny = pos(2); maxx = pos(1)+pos(3); maxy = pos(2)+pos(4);
                image_name = [patient_name, '-','IM',imagename(imglocs2(1)-4:imglocs2(1)-1),...
                    '_',time_name,'_',plaque_position,'_',position,'(',num2str(plaque_name),')','.bmp'];
                saveimagepath = fullfile(savepath, image_name);
                imwrite(image, saveimagepath);  
                
                frame=getframe(gcf);
                result=frame2im(frame);
                imwrite(result,fullfile(savepath_roi,image_name));

                fprintf(fid,'%s, %d, %d, %d, %d, %d\r\n', saveimagepath, minx,maxx,miny,maxy,resulution);           
                flag = input('done?(input 0 if done)');
                if flag ==0
                    break;
                end 
            end
            
        end    
    end
end

fclose(fid);
fclose(fid2);






