classdef ConvLayer1002 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        e_conv3_bias
        e_conv3_weight
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
            name = 'ZeroDCE.coder.ConvLayer1002';
        end
    end


    methods
        function this = ConvLayer1002(name)
            this.Name = name;
            this.OutputNames = {'conv2d_2'};
        end

        function [conv2d_2] = predict(this, relu_1)
            if isdlarray(relu_1)
                relu_1 = stripdims(relu_1);
            end
            relu_1NumDims = 4;
            relu_1 = ZeroDCE.ops.permuteInputVar(relu_1, [4 3 1 2], 4);

            [conv2d_2, conv2d_2NumDims] = ConvGraph1010(this, relu_1, relu_1NumDims, false);
            conv2d_2 = ZeroDCE.ops.permuteOutputVar(conv2d_2, [3 4 2 1], 4);

            conv2d_2 = dlarray(single(conv2d_2), 'SSCB');
        end

        function [conv2d_2] = forward(this, relu_1)
            if isdlarray(relu_1)
                relu_1 = stripdims(relu_1);
            end
            relu_1NumDims = 4;
            relu_1 = ZeroDCE.ops.permuteInputVar(relu_1, [4 3 1 2], 4);

            [conv2d_2, conv2d_2NumDims] = ConvGraph1010(this, relu_1, relu_1NumDims, true);
            conv2d_2 = ZeroDCE.ops.permuteOutputVar(conv2d_2, [3 4 2 1], 4);

            conv2d_2 = dlarray(single(conv2d_2), 'SSCB');
        end

        function [conv2d_2, conv2d_2NumDims1014] = ConvGraph1010(this, relu_1, relu_1NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights, bias, stride, dilationFactor, padding, dataFormat, conv2d_2NumDims] = ZeroDCE.ops.prepareConvArgs(this.e_conv3_weight, this.e_conv3_bias, this.Vars.ConvStride1011, this.Vars.ConvDilationFactor1012,this.Vars.ConvPadding1013, 1, relu_1NumDims, this.NumDims.e_conv3_weight);
            conv2d_2 = dlconv(relu_1, weights, bias, 'Stride', stride, 'DilationFactor', dilationFactor, 'Padding', padding, 'DataFormat', dataFormat);

            % Set graph output arguments
            conv2d_2NumDims1014 = conv2d_2NumDims;

        end

    end

end