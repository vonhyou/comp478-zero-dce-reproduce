classdef ConvLayer1005 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        e_conv6_bias
        e_conv6_weight
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
            name = 'ZeroDCE.coder.ConvLayer1005';
        end
    end


    methods
        function this = ConvLayer1005(name)
            this.Name = name;
            this.OutputNames = {'conv2d_5'};
        end

        function [conv2d_5] = predict(this, cat_1)
            if isdlarray(cat_1)
                cat_1 = stripdims(cat_1);
            end
            cat_1NumDims = 4;
            cat_1 = ZeroDCE.ops.permuteInputVar(cat_1, [4 3 1 2], 4);

            [conv2d_5, conv2d_5NumDims] = ConvGraph1025(this, cat_1, cat_1NumDims, false);
            conv2d_5 = ZeroDCE.ops.permuteOutputVar(conv2d_5, [3 4 2 1], 4);

            conv2d_5 = dlarray(single(conv2d_5), 'SSCB');
        end

        function [conv2d_5] = forward(this, cat_1)
            if isdlarray(cat_1)
                cat_1 = stripdims(cat_1);
            end
            cat_1NumDims = 4;
            cat_1 = ZeroDCE.ops.permuteInputVar(cat_1, [4 3 1 2], 4);

            [conv2d_5, conv2d_5NumDims] = ConvGraph1025(this, cat_1, cat_1NumDims, true);
            conv2d_5 = ZeroDCE.ops.permuteOutputVar(conv2d_5, [3 4 2 1], 4);

            conv2d_5 = dlarray(single(conv2d_5), 'SSCB');
        end

        function [conv2d_5, conv2d_5NumDims1029] = ConvGraph1025(this, cat_1, cat_1NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights, bias, stride, dilationFactor, padding, dataFormat, conv2d_5NumDims] = ZeroDCE.ops.prepareConvArgs(this.e_conv6_weight, this.e_conv6_bias, this.Vars.ConvStride1026, this.Vars.ConvDilationFactor1027,this.Vars.ConvPadding1028, 1, cat_1NumDims, this.NumDims.e_conv6_weight);
            conv2d_5 = dlconv(cat_1, weights, bias, 'Stride', stride, 'DilationFactor', dilationFactor, 'Padding', padding, 'DataFormat', dataFormat);

            % Set graph output arguments
            conv2d_5NumDims1029 = conv2d_5NumDims;

        end

    end

end