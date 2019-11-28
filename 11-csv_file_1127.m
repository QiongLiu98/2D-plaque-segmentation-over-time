clc
clear all
close all



path_image = '/home/ran/MyProjs/2DPlaqueSeg/graham_SPARC_fixedS/2Dplaqueimage';

csv = '/home/ran/MyProjs/2DPlaqueSeg/Datalist_2DPlaque_SPARCM_timepoint_1010.csv';
 
fid = fopen(csv,'a');
objects = dir(path_image);
for i = 3:length(objects)
    if rem(i,2)==1
        image_name = objects(i).name;
        plaque_name = objects(i+1).name;
    %plaque_name = '285136-IM0001_05-20-2014_M_RTCRO(1)-mask.tiff' ;
  
        saveimagepath = fullfile(path_image, image_name);
        savelabelpath = fullfile(path_image, plaque_name);
       

        fprintf(fid,'%s,%s\r\n', saveimagepath, savelabelpath);
    end
end

fclose(fid);





