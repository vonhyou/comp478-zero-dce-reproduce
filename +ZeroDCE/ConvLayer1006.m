classdef ConvLayer1006 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        e_conv7_bias
        e_conv7_weight
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
            name = 'ZeroDCE.coder.ConvLayer1006';
        end
    end


    methods
        function this = ConvLayer1006(name)
            this.Name = name;
            this.OutputNames = {'conv2d_6'};
        end

        function [conv2d_6] = predict(this, cat_2)
            if isdlarray(cat_2)
                cat_2 = stripdims(cat_2);
            end
            cat_2NumDims = 4;
            cat_2 = ZeroDCE.ops.permuteInputVar(cat_2, [4 3 1 2], 4);

            [conv2d_6, conv2d_6NumDims] = ConvGraph1030(this, cat_2, cat_2NumDims, false);
            conv2d_6 = ZeroDCE.ops.permuteOutputVar(conv2d_6, [3 4 2 1], 4);

            conv2d_6 = dlarray(single(conv2d_6), 'SSCB');
        end

        function [conv2d_6] = forward(this, cat_2)
            if isdlarray(cat_2)
                cat_2 = stripdims(cat_2);
            end
            cat_2NumDims = 4;
            cat_2 = ZeroDCE.ops.permuteInputVar(cat_2, [4 3 1 2], 4);

            [conv2d_6, conv2d_6NumDims] = ConvGraph1030(this, cat_2, cat_2NumDims, true);
            conv2d_6 = ZeroDCE.ops.permuteOutputVar(conv2d_6, [3 4 2 1], 4);

            conv2d_6 = dlarray(single(conv2d_6), 'SSCB');
        end

        function [conv2d_6, conv2d_6NumDims1034] = ConvGraph1030(this, cat_2, cat_2NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights, bias, stride, dilationFactor, padding, dataFormat, conv2d_6NumDims] = ZeroDCE.ops.prepareConvArgs(this.e_conv7_weight, this.e_conv7_bias, this.Vars.ConvStride1031, this.Vars.ConvDilationFactor1032,this.Vars.ConvPadding1033, 1, cat_2NumDims, this.NumDims.e_conv7_weight);
            conv2d_6 = dlconv(cat_2, weights, bias, 'Stride', stride, 'DilationFactor', dilationFactor, 'Padding', padding, 'DataFormat', dataFormat);

            % Set graph output arguments
            conv2d_6NumDims1034 = conv2d_6NumDims;

        end

    end

end