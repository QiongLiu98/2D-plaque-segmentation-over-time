clc
clear all
close all



path_mask = '/home/ran/2DPlaqueData_SPARC/PLAQUE_comparision_test';
path_image = '/home/ran/2DPlaqueData_SPARC/PLAQUE_comparision_test_images';

savepath = '/home/ran/MyProjs/Get2DSegData/1126/2DplaqueC';
csvout = '/home/ran/MyProjs/Get2DSegData/1126/DatalistC_1126.csv';
if exist(savepath,'dir')==0
    mkdir(savepath);
end
    
fid = fopen(csvout,'a');
fprintf(fid,'%s,%s\r\n','filepath','labelpath');
objects1 = dir(path_mask);
objects2 = dir(path_image);
for i = 3:length(objects1)
    plaque_name = objects1(i).name;
    %plaque_name = '285136-IM0001_05-20-2014_M_RTCRO(1)-mask.tiff' ;
    locs = strfind(plaque_name,'-');
    image_save_name = plaque_name(1:locs(4)-1);
    image_name = [image_save_name,'.bmp'];
    imagepath = fullfile(path_image,image_name);
    labelpath = fullfile(path_mask,plaque_name);
    image = imread(imagepath);
    mask = imread(labelpath);
    saveimagepath = fullfile(savepath, [image_save_name '.bmp']);
    savelabelpath = fullfile(savepath, [image_save_name '_mask.bmp']);
    imwrite(image, saveimagepath);
    imwrite(logical(rgb2gray(mask)), savelabelpath);  
    %imwrite(logical((mask)), savelabelpath); 
    fprintf(fid,'%s,%s\r\n', saveimagepath, savelabelpath);
end

fclose(fid);





