close all; clear all;clc;

% set the default colormap
set(groot,'DefaultFigureColormap',gray);

%% 仿真参数设置
datapath        = 'D:\project\Pro7-denseDL\data\simulation\data4\bounledTubes_mySimu\simuTrainData';   % 数据路径

density = 1.0;

fileDensity = 200;
    tiff_filename   = sprintf('dataset_ROI64_F1000_density%.1f.tif',fileDensity);        % 原始图像
    csv_filename    = sprintf('dataset_ROI64_F1000_density%.1f.csv',fileDensity);        % groundtruth
    savePath        = [datapath '\' tiff_filename(1:end-7) sprintf('%.1f',fileDensity)];

    addpath(genpath(datapath));

    camera_pixelsize    = 100;                          % simulated camera pixel size in [nm]
    upsampling_factor   = 8;   

    minEmitters     = 1;        % 单张图的patch中，分子最小数量
    maxExamples     = 10000;    % 所有图像中提取，训练 pairs 最大数量
    gaussian_sigma  = 1;        % heatmap渲染用，标准差 [pixels]

    %% 图像读取
    ImageStack      = ReadStackFromTiff(tiff_filename); 
    [M,N,numImages] = size(ImageStack);    

    %% csv文件读取
    Activations = importdata(fullfile(datapath,csv_filename));
    Data        = Activations.data;
    col_names   = Activations.colheaders;


    pixelSize_hr    = camera_pixelsize/upsampling_factor;           % nm    % 渲染像素大小
    psfHeatmap      = fspecial('gauss',[7 7],gaussian_sigma);       % 渲染图每个点的psf % heatmap psf

    Mhr = upsampling_factor*M;                  % 超分辨图 行        
    Nhr = upsampling_factor*N;                  % 超分辨图 列 

    %% 判断图像中分子数量
    for frmNum  =   1:numImages
        % 原始图
        rawImg   = ImageStack(:,:,frmNum);
        DataFrame = Data(Data(:,2)==frmNum,:);                                          % 读取当前帧中定位分子

        % 超分辨图索引
        Chr_emitters = max(min(round(DataFrame(:,3)/pixelSize_hr),Nhr),1);              % 超分辨图中的索引
        Rhr_emitters = max(min(round(DataFrame(:,4)/pixelSize_hr),Mhr),1);

        if size(DataFrame,1) <  density
            continue;
        else
            rawImgUp = imresize(rawImg,upsampling_factor,'box');
            % spike image 和 heatmap image的关系
            indEmitters = sub2ind([Mhr,Nhr],Rhr_emitters,Chr_emitters);
            SpikesImage                                 = zeros(Mhr,Nhr);                       % 在索引处设置值为1，并卷积一个psfHeatmap
            SpikesImage(indEmitters) = 1;
            HeatmapImage                                = conv2(SpikesImage,psfHeatmap,'same'); % HeatmapImage是spikesImage的渲染图

            if ~exist(savePath)
                mkdir(savePath)
            end
            if ~exist(strcat(savePath,'/','rawImg'))
                mkdir(strcat(savePath,'/','rawImg'))
            end
            if ~exist(strcat(savePath,'/','rawImgUp'))
                mkdir(strcat(savePath,'/','rawImgUp'))
            end
            if ~exist(strcat(savePath,'/','HeatmapImg'))
                mkdir(strcat(savePath,'/','HeatmapImg'))
            end
            if ~exist(strcat(savePath,'/','SpikesImg'))
                mkdir(strcat(savePath,'/','SpikesImg'))
            end

            imwrite(rawImg,strcat(savePath,'/','rawImg','/',sprintf('%d.tif',frmNum)));
            imwrite(rawImgUp,strcat(savePath,'/','rawImgUp','/',sprintf('%d.tif',frmNum)));
            singleTiff_(single(HeatmapImage),strcat(savePath,'/','HeatmapImg','/',sprintf('%d.tif',frmNum)));
            singleTiff_(single(SpikesImage) ,strcat(savePath,'/','SpikesImg' ,'/',sprintf('%d.tif',frmNum)));
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
