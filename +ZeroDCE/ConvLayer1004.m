classdef ConvLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        e_conv5_bias
        e_conv5_weight
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
            name = 'ZeroDCE.coder.ConvLayer1004';
        end
    end


    methods
        function this = ConvLayer1004(name)
            this.Name = name;
            this.OutputNames = {'conv2d_4'};
        end

        function [conv2d_4] = predict(this, cat1000)
            if isdlarray(cat1000)
                cat1000 = stripdims(cat1000);
            end
            cat1000NumDims = 4;
            cat1000 = ZeroDCE.ops.permuteInputVar(cat1000, [4 3 1 2], 4);

            [conv2d_4, conv2d_4NumDims] = ConvGraph1020(this, cat1000, cat1000NumDims, false);
            conv2d_4 = ZeroDCE.ops.permuteOutputVar(conv2d_4, [3 4 2 1], 4);

            conv2d_4 = dlarray(single(conv2d_4), 'SSCB');
        end

        function [conv2d_4] = forward(this, cat1000)
            if isdlarray(cat1000)
                cat1000 = stripdims(cat1000);
            end
            cat1000NumDims = 4;
            cat1000 = ZeroDCE.ops.permuteInputVar(cat1000, [4 3 1 2], 4);

            [conv2d_4, conv2d_4NumDims] = ConvGraph1020(this, cat1000, cat1000NumDims, true);
            conv2d_4 = ZeroDCE.ops.permuteOutputVar(conv2d_4, [3 4 2 1], 4);

            conv2d_4 = dlarray(single(conv2d_4), 'SSCB');
        end

        function [conv2d_4, conv2d_4NumDims1024] = ConvGraph1020(this, cat1000, cat1000NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights, bias, stride, dilationFactor, padding, dataFormat, conv2d_4NumDims] = ZeroDCE.ops.prepareConvArgs(this.e_conv5_weight, this.e_conv5_bias, this.Vars.ConvStride1021, this.Vars.ConvDilationFactor1022,this.Vars.ConvPadding1023, 1, cat1000NumDims, this.NumDims.e_conv5_weight);
            conv2d_4 = dlconv(cat1000, weights, bias, 'Stride', stride, 'DilationFactor', dilationFactor, 'Padding', padding, 'DataFormat', dataFormat);

            % Set graph output arguments
            conv2d_4NumDims1024 = conv2d_4NumDims;

        end

    end

end