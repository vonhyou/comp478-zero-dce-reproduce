classdef ConvLayer1005 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                'e_conv6_bias'
                'e_conv6_weight'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = ZeroDCE.coder.ConvLayer1005(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ZeroDCE.ConvLayer1005(cgInstance.Name);
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
            this_ml.e_conv6_bias = cgInstance.e_conv6_bias;
            this_ml.e_conv6_weight = cgInstance.e_conv6_weight;
        end
    end

    methods
        function this = ConvLayer1005(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'conv2d_5'};
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
            this.e_conv6_bias = mlInstance.e_conv6_bias;
            this.e_conv6_weight = mlInstance.e_conv6_weight;
        end

        function [conv2d_5] = predict(this, cat_1__)
            if isdlarray(cat_1__)
                cat_1_ = stripdims(cat_1__);
            else
                cat_1_ = cat_1__;
            end
            cat_1NumDims = 4;
            cat_1 = ZeroDCE.coder.ops.permuteInputVar(cat_1_, [4 3 1 2], 4);

            [conv2d_5__, conv2d_5NumDims__] = ConvGraph1025(this, cat_1, cat_1NumDims, false);
            conv2d_5_ = ZeroDCE.coder.ops.permuteOutputVar(conv2d_5__, [3 4 2 1], 4);

            conv2d_5 = dlarray(single(conv2d_5_), 'SSCB');
        end

        function [conv2d_5, conv2d_5NumDims1029] = ConvGraph1025(this, cat_1, cat_1NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights1040, bias1041, stride1042, dilationFactor1043, padding1044, dataFormat1045, conv2d_5NumDims] = ZeroDCE.coder.ops.prepareConvArgs(this.e_conv6_weight, this.e_conv6_bias, this.Vars.ConvStride1026, this.Vars.ConvDilationFactor1027,this.Vars.ConvPadding1028, 1, cat_1NumDims, this.NumDims.e_conv6_weight);
            X1046 = dlarray(single(ZeroDCE.coder.ops.extractIfDlarray(cat_1)));
            Y1047 = dlconv(X1046, weights1040, bias1041, 'Stride', stride1042, 'DilationFactor', dilationFactor1043, 'Padding', padding1044, 'DataFormat', dataFormat1045);
            conv2d_5 = ZeroDCE.coder.ops.extractIfDlarray(Y1047);

            % Set graph output arguments
            conv2d_5NumDims1029 = conv2d_5NumDims;

        end

    end

end