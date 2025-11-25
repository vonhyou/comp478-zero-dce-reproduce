function x = apply_zerodce(x, paramsMap, iterations)
% APPLY_ZERODCE Applies the iterative curve enhancement formula
% Input:
%   x: The input dlarray image (SSCB)
%   paramsMap: The estimated curve parameters (24 channels)
    
    for i = 1:iterations
        % Extract alpha map for the current iteration
        idx_start = (i-1)*3 + 1;
        idx_end = i*3;
        alpha_n = paramsMap(:, :, idx_start:idx_end, :);
        
        % Apply Zero-DCE curve equation
        % Formula: x = x + alpha * (x^2 - x)
        x = x + alpha_n .* (x .^ 2 - x);
    end
end