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
    rawImg_uint8 = rawImg;
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
iterations = 8;
x = apply_zerodce(dlInput, paramsMap, iterations);

%% 5. Post-Processing
enhancedImg = extractdata(x);
enhancedImg = min(max(enhancedImg, 0), 1);
enhancedImg_uint8 = im2uint8(enhancedImg);

%% 6. Quality Analysis (NIQE & PIQE)
disp('Calculating Quality Metrics (Lower is Better)...');

% Calculate NIQE
score_niqe_input = niqe(rawImg_uint8);
score_niqe_output = niqe(enhancedImg_uint8);

% Calculate PIQE
score_piqe_input = piqe(rawImg_uint8);
score_piqe_output = piqe(enhancedImg_uint8);

fprintf('\n--- Quality Report ---\n');
fprintf('NIQE (Naturalness): Input = %.4f | Output = %.4f \n', score_niqe_input, score_niqe_output);
fprintf('PIQE (Perception) : Input = %.4f | Output = %.4f \n', score_piqe_input, score_piqe_output);

%% 7. Visualization
figure('Name', 'Zero-DCE MATLAB Result');
subplot(1,2,1); 
imshow(rawImg); 
title(sprintf('Input\nNIQE: %.2f | PIQE: %.2f', score_niqe_input, score_piqe_input));

subplot(1,2,2); 
imshow(enhancedImg); 
title(sprintf('Enhanced\nNIQE: %.2f | PIQE: %.2f', score_niqe_output, score_piqe_output));

disp('Done.');