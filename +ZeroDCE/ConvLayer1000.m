classdef ConvLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        e_conv1_bias
        e_conv1_weight
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end


    methods(Static, Hidden)
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'ZeroDCE.coder.ConvLayer1000';
        end
    end


    methods
        function this = ConvLayer1000(name)
            this.Name = name;
            this.OutputNames = {'conv2d'};
        end

        function [conv2d] = predict(this, input_image)
            if isdlarray(input_image)
                input_image = stripdims(input_image);
            end
            input_imageNumDims = 4;
            input_image = ZeroDCE.ops.permuteInputVar(input_image, [4 3 1 2], 4);

            [conv2d, conv2dNumDims] = ConvGraph1000(this, input_image, input_imageNumDims, false);
            conv2d = ZeroDCE.ops.permuteOutputVar(conv2d, [3 4 2 1], 4);

            conv2d = dlarray(single(conv2d), 'SSCB');
        end

        function [conv2d] = forward(this, input_image)
            if isdlarray(input_image)
                input_image = stripdims(input_image);
            end
            input_imageNumDims = 4;
            input_image = ZeroDCE.ops.permuteInputVar(input_image, [4 3 1 2], 4);

            [conv2d, conv2dNumDims] = ConvGraph1000(this, input_image, input_imageNumDims, true);
            conv2d = ZeroDCE.ops.permuteOutputVar(conv2d, [3 4 2 1], 4);

            conv2d = dlarray(single(conv2d), 'SSCB');
        end

        function [conv2d, conv2dNumDims1004] = ConvGraph1000(this, input_image, input_imageNumDims, Training)

            % Execute the operators:
            % Conv:
            [weights, bias, stride, dilationFactor, padding, dataFormat, conv2dNumDims] = ZeroDCE.ops.prepareConvArgs(this.e_conv1_weight, this.e_conv1_bias, this.Vars.ConvStride1001, this.Vars.ConvDilationFactor1002,this.Vars.ConvPadding1003, 1, input_imageNumDims, this.NumDims.e_conv1_weight);
            conv2d = dlconv(input_image, weights, bias, 'Stride', stride, 'DilationFactor', dilationFactor, 'Padding', padding, 'DataFormat', dataFormat);

            % Set graph output arguments
            conv2dNumDims1004 = conv2dNumDims;

        end

    end

end