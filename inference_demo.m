close all; clear; clc;

%% 1. Load the ONNX
modelFile = 'models/ZeroDCE.onnx';
if ~exist(modelFile, 'file')
    error('File %s not found.', modelFile);
end
disp('Load ONNX model...');
net = importNetworkFromONNX(modelFile);

%% 2. Load and Preprocess Image
imageFile = './images/input/MYOWN/006.jpg'; 
if ~exist(imageFile, 'file')
    error('Image %s not found.', imageFile);
else
    rawImg = imread(imageFile);
    rawImg = im2single(rawImg);
end

% Prepare input for network (Height x Width x Channel x Batch)
dlInput = dlarray(rawImg, 'SSCB');
if ndims(dlInput) < 4
    dlInput = expanddims(dlInput, 4);
end

disp('Initializing network...');
net = initialize(net, dlInput);

%% 3. Predict Curve Parameters
disp('Predicting curve parameters...');
paramsMap = predict(net, dlInput); 

%% 4. Apply Enhancement Loop
% The network outputs 24 channels (8 iterations * 3 channels)
x = dlInput; 
iterations = 8;

disp('Applying iterative curve enhancement...');
for i = 1:iterations
    % Extract alpha map for the current iteration
    idx_start = (i-1)*3 + 1;
    idx_end = i*3;
    alpha_n = paramsMap(:, :, idx_start:idx_end, :);
    
    % Apply Zero-DCE curve equation
    x = x + alpha_n .* (x .^ 2 - x);
end

%% 5. Post-Processing and Visualization
enhancedImg = extractdata(x);
enhancedImg = min(max(enhancedImg, 0), 1);

figure('Name', 'Zero-DCE MATLAB Result');
subplot(1,2,1); imshow(rawImg); title('Input');
subplot(1,2,2); imshow(enhancedImg); title('Enhanced');
disp('Done.');