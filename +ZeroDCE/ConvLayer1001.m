classdef ConvLayer1001 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        e_conv2_bias
        e_conv2_weight
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
            name = 'ZeroDCE.coder.ConvLayer1001';
        end
    end


    methods
        function this = ConvLayer1001(name)
            this.Name = name;
            this.OutputNames = {'conv2d_1'};
        end

        function [conv2d_1] = predict(this, relu1001)
            if isdlarray(relu1001)
                relu1001 = stripdims(relu1001);
            end
            relu1001NumDims = 4;
            relu1001 = ZeroDCE.ops.permuteInputVar(relu1001, [4 3 1 2], 4);

            [conv2d_1, conv2d_1NumDims] = ConvGraph1005(this, relu1001, relu1001NumDims, false);
            conv2d_1 = ZeroDCE.ops.permuteOutputVar(conv2d_1, [3 4 2 1], 4);

            conv2d_1 = dlarray(single(conv2d_1), 'SSCB');
        end

        function [conv2d_1] = forward(this, relu1001)
            if isdlarray(relu1001)
                relu1001 = stripdims(relu1001);
            end
            relu1001NumDims = 4;
            relu1001 = ZeroDCE.ops.permuteInputVar(relu1001, [4 3 1 2], 4);

            [conv2d_1, conv2d_1NumDims] = ConvGraph1005(this, relu1001, relu1001NumDims, true);
            conv2d_1 = ZeroDCE.ops.permuteOutputVar(conv2d_1, [3 4 2 1], 4);

            conv2d_1 = dlarray(single(conv2d_1), 'SSCB');
        end

        function [conv2d_1, conv2d_1NumDims1009] = ConvGraph1005(this, relu1001, relu1001NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights, bias, stride, dilationFactor, padding, dataFormat, conv2d_1NumDims] = ZeroDCE.ops.prepareConvArgs(this.e_conv2_weight, this.e_conv2_bias, this.Vars.ConvStride1006, this.Vars.ConvDilationFactor1007,this.Vars.ConvPadding1008, 1, relu1001NumDims, this.NumDims.e_conv2_weight);
            conv2d_1 = dlconv(relu1001, weights, bias, 'Stride', stride, 'DilationFactor', dilationFactor, 'Padding', padding, 'DataFormat', dataFormat);

            % Set graph output arguments
            conv2d_1NumDims1009 = conv2d_1NumDims;

        end

    end

end