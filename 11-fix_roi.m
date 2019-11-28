clc
clear all;
close all;
path_image = '/home/ran/2DPlaqueData_SPARC/US-carotid-images-multiple-timepoints/11';
savepath = './1121_roi/2DplaqueQ';
savepath_roi = './1121_roi/2DplaqueQ_with_ROI';
csvout = './1120_roi/Datalist_qiong_1113.csv';
fid = fopen(csvout, 'a');
%995279-IM0008_04-27-2016_R_LTICA(1).bmp
%fixed image name
patient_name = '995279';
image_name_ori = '04-27-20160008.jpg';
% doc_area = 59;
resolution = 0.065;
imglocs2 = strfind(image_name_ori,'.');
time_name = image_name_ori(1:imglocs2(1)-5);
%mth = {'Jan', 'Feb', 'Mar', 'Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
imagepath = fullfile(path_image,patient_name,time_name,image_name_ori);

image = imread(imagepath);
im = imshow(image);
title(imagepath)
mouse=imrect;

position = input ('position(eg:RTICA):','s');
plaque_position = input('l/r/m):','s');
plaque_name = input('plaque_name(eg:1/2/3):');


pos=getPosition(mouse);% x1 y1 w h
minx = pos(1); miny = pos(2); maxx = pos(1)+pos(3); maxy = pos(2)+pos(4);
image_name = [patient_name, '-','IM',image_name_ori(imglocs2(1)-4:imglocs2(1)-1),...
    '_',time_name,'_',plaque_position,'_',position,'(',num2str(plaque_name),')','.bmp'];
saveimagepath = fullfile(savepath, image_name);
imwrite(image, saveimagepath);  

frame=getframe(gcf);
result=frame2im(frame);
imwrite(result,fullfile(savepath_roi,image_name));

fprintf(fid,'%s, %d, %d, %d, %d, %d\r\n', saveimagepath, minx,maxx,miny,maxy,resolution);           

           

fclose(fid);






