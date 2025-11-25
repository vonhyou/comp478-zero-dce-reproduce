classdef ConvLayer1001 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                'e_conv2_bias'
                'e_conv2_weight'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = ZeroDCE.coder.ConvLayer1001(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ZeroDCE.ConvLayer1001(cgInstance.Name);
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
            this_ml.e_conv2_bias = cgInstance.e_conv2_bias;
            this_ml.e_conv2_weight = cgInstance.e_conv2_weight;
        end
    end

    methods
        function this = ConvLayer1001(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'conv2d_1'};
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
            this.e_conv2_bias = mlInstance.e_conv2_bias;
            this.e_conv2_weight = mlInstance.e_conv2_weight;
        end

        function [conv2d_1] = predict(this, relu1001__)
            if isdlarray(relu1001__)
                relu1001_ = stripdims(relu1001__);
            else
                relu1001_ = relu1001__;
            end
            relu1001NumDims = 4;
            relu1001 = ZeroDCE.coder.ops.permuteInputVar(relu1001_, [4 3 1 2], 4);

            [conv2d_1__, conv2d_1NumDims__] = ConvGraph1005(this, relu1001, relu1001NumDims, false);
            conv2d_1_ = ZeroDCE.coder.ops.permuteOutputVar(conv2d_1__, [3 4 2 1], 4);

            conv2d_1 = dlarray(single(conv2d_1_), 'SSCB');
        end

        function [conv2d_1, conv2d_1NumDims1009] = ConvGraph1005(this, relu1001, relu1001NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights1008, bias1009, stride1010, dilationFactor1011, padding1012, dataFormat1013, conv2d_1NumDims] = ZeroDCE.coder.ops.prepareConvArgs(this.e_conv2_weight, this.e_conv2_bias, this.Vars.ConvStride1006, this.Vars.ConvDilationFactor1007,this.Vars.ConvPadding1008, 1, relu1001NumDims, this.NumDims.e_conv2_weight);
            X1014 = dlarray(single(ZeroDCE.coder.ops.extractIfDlarray(relu1001)));
            Y1015 = dlconv(X1014, weights1008, bias1009, 'Stride', stride1010, 'DilationFactor', dilationFactor1011, 'Padding', padding1012, 'DataFormat', dataFormat1013);
            conv2d_1 = ZeroDCE.coder.ops.extractIfDlarray(Y1015);

            % Set graph output arguments
            conv2d_1NumDims1009 = conv2d_1NumDims;

        end

    end

end