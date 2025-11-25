classdef ConvLayer1002 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                'e_conv3_bias'
                'e_conv3_weight'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = ZeroDCE.coder.ConvLayer1002(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ZeroDCE.ConvLayer1002(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
            this_ml.e_conv3_bias = cgInstance.e_conv3_bias;
            this_ml.e_conv3_weight = cgInstance.e_conv3_weight;
        end
    end

    methods
        function this = ConvLayer1002(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'conv2d_2'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = ZeroDCE.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
            this.e_conv3_bias = mlInstance.e_conv3_bias;
            this.e_conv3_weight = mlInstance.e_conv3_weight;
        end

        function [conv2d_2] = predict(this, relu_1__)
            if isdlarray(relu_1__)
                relu_1_ = stripdims(relu_1__);
            else
                relu_1_ = relu_1__;
            end
            relu_1NumDims = 4;
            relu_1 = ZeroDCE.coder.ops.permuteInputVar(relu_1_, [4 3 1 2], 4);

            [conv2d_2__, conv2d_2NumDims__] = ConvGraph1010(this, relu_1, relu_1NumDims, false);
            conv2d_2_ = ZeroDCE.coder.ops.permuteOutputVar(conv2d_2__, [3 4 2 1], 4);

            conv2d_2 = dlarray(single(conv2d_2_), 'SSCB');
        end

        function [conv2d_2, conv2d_2NumDims1014] = ConvGraph1010(this, relu_1, relu_1NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights1016, bias1017, stride1018, dilationFactor1019, padding1020, dataFormat1021, conv2d_2NumDims] = ZeroDCE.coder.ops.prepareConvArgs(this.e_conv3_weight, this.e_conv3_bias, this.Vars.ConvStride1011, this.Vars.ConvDilationFactor1012,this.Vars.ConvPadding1013, 1, relu_1NumDims, this.NumDims.e_conv3_weight);
            X1022 = dlarray(single(ZeroDCE.coder.ops.extractIfDlarray(relu_1)));
            Y1023 = dlconv(X1022, weights1016, bias1017, 'Stride', stride1018, 'DilationFactor', dilationFactor1019, 'Padding', padding1020, 'DataFormat', dataFormat1021);
            conv2d_2 = ZeroDCE.coder.ops.extractIfDlarray(Y1023);

            % Set graph output arguments
            conv2d_2NumDims1014 = conv2d_2NumDims;

        end

    end

end