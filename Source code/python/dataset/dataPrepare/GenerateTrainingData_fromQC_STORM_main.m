close all; clear all;  clc;

% set the default colormap
set(groot,'DefaultFigureColormap',gray);

%% parameters setting
datapath        = '..\data';   % data path
overlapFactor   = 2;           % the nums of raw images that are overlaped

density         = 1.0;         % filter, the density below 1.0 of raw image will be remove
tiff_filename   = strcat(datapath,'\','rawImg.tif');                        % raw image stack, .tif
txt_filename    = strcat(datapath,'\','rawImg_result2D7_M.txt');            % groundtruth from qc-storm
savePath        = [datapath '\' 'result'];                                  % the result directory

camera_pixelsize    = 100;      % camera pixel size in [nm]
upsampling_factor   = 8;        % upsampling factor, raw image will be upsampled x(factor) times
kernelSize          = 7;        % kernel size, 

gaussian_sigma      = 1;            % using for heatmap，standard error, unit is pixels

addpath(genpath(datapath));
%% 图像读取
ImageStack      = ReadStackFromTiff(tiff_filename); 
[M,N,numImages] = size(ImageStack);    

%% csv文件读取
Data = loadQCtxt(txt_filename);

pixelSize_hr    = camera_pixelsize/upsampling_factor;           % nm    % 渲染像素大小
psfHeatmap      = fspecial('gauss',[kernelSize kernelSize],gaussian_sigma);       % 渲染图每个点的psf % heatmap psf

Mhr = upsampling_factor*M;                  % 超分辨图 行        
Nhr = upsampling_factor*N;                  % 超分辨图 列 

%% 判断图像中分子数量
for frmNum  =   1:overlapFactor:numImages
    % 原始图
    rawImg    = mean(single(ImageStack(:,:,frmNum:(frmNum+overlapFactor-1))),3);
    DataFrame = Data(Data(:,12)>=frmNum & Data(:,12)< frmNum+overlapFactor  ,:);                                          % 读取当前帧中定位分子

    % 超分辨图索引
    Chr_emitters = max(min(round(DataFrame(:,2)*camera_pixelsize/pixelSize_hr),Nhr),1);              % 超分辨图中的索引
    Rhr_emitters = max(min(round(DataFrame(:,3)*camera_pixelsize/pixelSize_hr),Mhr),1);

    if size(DataFrame,1) <  density
        continue;
    else
%         rawImgUp = imresize(rawImg,upsampling_factor,'box');
        % spike image 和 heatmap image的关系
        indEmitters = sub2ind([Mhr,Nhr],Rhr_emitters,Chr_emitters);
        SpikesImage                                 = zeros(Mhr,Nhr);                       % 在索引处设置值为1，并卷积一个psfHeatmap
        SpikesImage(indEmitters)                    = 1;
        HeatmapImage                                = conv2(SpikesImage,psfHeatmap,'same'); % HeatmapImage是spikesImage的渲染图

        if ~exist(savePath)
            mkdir(savePath)
        end
        if ~exist(strcat(savePath,'/','rawImg'))
            mkdir(strcat(savePath,'/','rawImg'))
        end
%         if ~exist(strcat(savePath,'/','rawImgUp'))
%             mkdir(strcat(savePath,'/','rawImgUp'))
%         end
        if ~exist(strcat(savePath,'/','HeatmapImg'))
            mkdir(strcat(savePath,'/','HeatmapImg'))
        end
%             if ~exist(strcat(savePath,'/','SpikesImg'))
%                 mkdir(strcat(savePath,'/','SpikesImg'))
%             end

        singleTiff_(rawImg,strcat(savePath,'/','rawImg','/',sprintf('%d.tif',frmNum)));
%         singleTiff_(rawImgUp,strcat(savePath,'/','rawImgUp','/',sprintf('%d.tif',frmNum)));
        singleTiff_(single(HeatmapImage),strcat(savePath,'/','HeatmapImg','/',sprintf('%d.tif',frmNum)));
%             singleTiff_(single(SpikesImage) ,strcat(savePath,'/','SpikesImg' ,'/',sprintf('%d.tif',frmNum)));
    end
    fprintf('total frame:%d,current frame:%d\n',numImages,frmNum)
end

function singleTiff_(input,outPutFilePath)
    %如果 BitPerSample 为 32，则输入图像数据类型必须为 int32、uint32 或单精度，而不是 double
    % This is a direct interface to libtiff
    t = Tiff(outPutFilePath,'w');

    % Setup tags
    % Lots of info here:
    % http://www.mathworks.com/help/matlab/ref/tiffclass.html
	% 参考matlab help Tiff
    tagstruct.ImageLength     = size(input,1);
    tagstruct.ImageWidth      = size(input,2);
    tagstruct.SampleFormat    = Tiff.SampleFormat.IEEEFP;
    tagstruct.BitsPerSample   = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
    tagstruct.RowsPerStrip    = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software        = 'MATLAB';
    t.setTag(tagstruct)

    t.write(input);
    t.close();  
end
