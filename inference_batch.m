close all; clear; clc;
%% 1. Configuration and Model Loading
inputRootDir = './images/input';
outputRootDir = './images/output';
reportFile = 'Report.md';
modelFile = 'models/ZeroDCE.onnx';
iterations = 8;

if ~exist(modelFile, 'file')
    error('File %s not found.', modelFile);
end

disp('Load ONNX model...');
netBase = importNetworkFromONNX(modelFile);

% Initialize Report File
fid = fopen(reportFile, 'w');
fprintf(fid, '# Zero-DCE Batch Report\n');
fprintf(fid, 'Date: %s\n\n', datetime("now"));

%% 2. Iterate Over Datasets
items = dir(inputRootDir);
% Filter only real directories (exclude . and ..)
subFolders = items([items.isdir] & ~strcmp({items.name},'.') & ~strcmp({items.name},'..'));

for k = 1:length(subFolders)
    datasetName = subFolders(k).name;
    datasetInDir = fullfile(inputRootDir, datasetName);
    datasetOutDir = fullfile(outputRootDir, datasetName);
    
    % Create output directory if needed
    if ~exist(datasetOutDir, 'dir')
        mkdir(datasetOutDir);
    end
    
    % Find valid images
    allItems = dir(fullfile(datasetInDir, '*'));
    fileNames = {allItems.name};
    isImage = false(size(allItems));
    validExts = {'.jpg', '.jpeg', '.png', '.bmp'};
    
    for m = 1:length(allItems)
        if allItems(m).isdir
            continue;
        end
        [~, ~, ext] = fileparts(allItems(m).name);
        if any(strcmpi(ext, validExts))
            isImage(m) = true;
        end
    end
    imgFiles = allItems(isImage);
    
    numImages = length(imgFiles);
    fprintf('Processing Dataset: %s (%d images)...\n', datasetName, numImages);
    
    % Stats accumulators
    stats.niqe_in = 0; stats.niqe_out = 0;
    stats.piqe_in = 0; stats.piqe_out = 0;
    stats.proc_time = 0;

    %% 3. Process Images
    hWait = waitbar(0, sprintf('Processing %s...', datasetName));
    for j = 1:numImages
        imgName = imgFiles(j).name;
        imageFile = fullfile(datasetInDir, imgName);
        outFile = fullfile(datasetOutDir, imgName);
        
        try
            rawImg = imread(imageFile);
            rawImg_uint8 = rawImg;
            rawImg = im2single(rawImg);
            
            % Prepare input for network (Height x Width x Channel x Batch)
            dlInput = dlarray(rawImg, 'SSCB');
            if ndims(dlInput) < 4
                dlInput = expanddims(dlInput, 4);
            end
            
            % Initialize network (Must do per-image for dynamic sizes)
            net = initialize(netBase, dlInput);

            t_img = tic;
            
            % Predict Curve Parameters
            paramsMap = predict(net, dlInput); 
            
            % Apply Enhancement Loop (Using Function)
            x = apply_zerodce(dlInput, paramsMap, iterations);

            elapsed = toc(t_img);
            stats.proc_time = stats.proc_time + elapsed;
            
            % Post-Processing
            enhancedImg = extractdata(x);
            enhancedImg = min(max(enhancedImg, 0), 1);
            enhancedImg_uint8 = im2uint8(enhancedImg);
            
            % Save Output
            imwrite(enhancedImg_uint8, outFile);
            
            % Calculate Metrics
            stats.niqe_in = stats.niqe_in + niqe(rawImg_uint8);
            stats.niqe_out = stats.niqe_out + niqe(enhancedImg_uint8);
            stats.piqe_in = stats.piqe_in + piqe(rawImg_uint8);
            stats.piqe_out = stats.piqe_out + piqe(enhancedImg_uint8);
            
        catch ME
            fprintf('\n'); 
            warning('Failed to process %s: %s', imgName, ME.message);
        end

        waitbar(j/numImages, hWait);
    end

    close(hWait);
    
    %% 4. Append to Report
    if numImages > 0
        avg_niqe_in = stats.niqe_in / numImages;
        avg_niqe_out = stats.niqe_out / numImages;
        avg_piqe_in = stats.piqe_in / numImages;
        avg_piqe_out = stats.piqe_out / numImages;
        avg_time = stats.proc_time / numImages;
        
        fprintf(fid, '## Dataset: %s\n', datasetName);
        fprintf(fid, '> number of images: %d\n\n', numImages);
        fprintf(fid, '> average inference time: %.4f seconds/image\n\n', avg_time);
        fprintf(fid, '| Metric | Original (Avg) | Enhanced (Avg) |\n');
        fprintf(fid, '| :--- | :---: | :---: |\n');
        fprintf(fid, '| **NIQE** | %.4f | %.4f |\n', avg_niqe_in, avg_niqe_out);
        fprintf(fid, '| **PIQE** | %.4f | %.4f |\n\n', avg_piqe_in, avg_piqe_out);
    end
end

fclose(fid);
disp('Done. Report saved to Report.md');