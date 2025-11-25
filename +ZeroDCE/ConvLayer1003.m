classdef ConvLayer1003 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        e_conv4_bias
        e_conv4_weight
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
            name = 'ZeroDCE.coder.ConvLayer1003';
        end
    end


    methods
        function this = ConvLayer1003(name)
            this.Name = name;
            this.OutputNames = {'conv2d_3'};
        end

        function [conv2d_3] = predict(this, relu_2)
            if isdlarray(relu_2)
                relu_2 = stripdims(relu_2);
            end
            relu_2NumDims = 4;
            relu_2 = ZeroDCE.ops.permuteInputVar(relu_2, [4 3 1 2], 4);

            [conv2d_3, conv2d_3NumDims] = ConvGraph1015(this, relu_2, relu_2NumDims, false);
            conv2d_3 = ZeroDCE.ops.permuteOutputVar(conv2d_3, [3 4 2 1], 4);

            conv2d_3 = dlarray(single(conv2d_3), 'SSCB');
        end

        function [conv2d_3] = forward(this, relu_2)
            if isdlarray(relu_2)
                relu_2 = stripdims(relu_2);
            end
            relu_2NumDims = 4;
            relu_2 = ZeroDCE.ops.permuteInputVar(relu_2, [4 3 1 2], 4);

            [conv2d_3, conv2d_3NumDims] = ConvGraph1015(this, relu_2, relu_2NumDims, true);
            conv2d_3 = ZeroDCE.ops.permuteOutputVar(conv2d_3, [3 4 2 1], 4);

            conv2d_3 = dlarray(single(conv2d_3), 'SSCB');
        end

        function [conv2d_3, conv2d_3NumDims1019] = ConvGraph1015(this, relu_2, relu_2NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights, bias, stride, dilationFactor, padding, dataFormat, conv2d_3NumDims] = ZeroDCE.ops.prepareConvArgs(this.e_conv4_weight, this.e_conv4_bias, this.Vars.ConvStride1016, this.Vars.ConvDilationFactor1017,this.Vars.ConvPadding1018, 1, relu_2NumDims, this.NumDims.e_conv4_weight);
            conv2d_3 = dlconv(relu_2, weights, bias, 'Stride', stride, 'DilationFactor', dilationFactor, 'Padding', padding, 'DataFormat', dataFormat);

            % Set graph output arguments
            conv2d_3NumDims1019 = conv2d_3NumDims;

        end

    end

end