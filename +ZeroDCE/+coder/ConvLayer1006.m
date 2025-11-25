classdef ConvLayer1006 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                'e_conv7_bias'
                'e_conv7_weight'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = ZeroDCE.coder.ConvLayer1006(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ZeroDCE.ConvLayer1006(cgInstance.Name);
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
            this_ml.e_conv7_bias = cgInstance.e_conv7_bias;
            this_ml.e_conv7_weight = cgInstance.e_conv7_weight;
        end
    end

    methods
        function this = ConvLayer1006(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'conv2d_6'};
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
            this.e_conv7_bias = mlInstance.e_conv7_bias;
            this.e_conv7_weight = mlInstance.e_conv7_weight;
        end

        function [conv2d_6] = predict(this, cat_2__)
            if isdlarray(cat_2__)
                cat_2_ = stripdims(cat_2__);
            else
                cat_2_ = cat_2__;
            end
            cat_2NumDims = 4;
            cat_2 = ZeroDCE.coder.ops.permuteInputVar(cat_2_, [4 3 1 2], 4);

            [conv2d_6__, conv2d_6NumDims__] = ConvGraph1030(this, cat_2, cat_2NumDims, false);
            conv2d_6_ = ZeroDCE.coder.ops.permuteOutputVar(conv2d_6__, [3 4 2 1], 4);

            conv2d_6 = dlarray(single(conv2d_6_), 'SSCB');
        end

        function [conv2d_6, conv2d_6NumDims1034] = ConvGraph1030(this, cat_2, cat_2NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights1048, bias1049, stride1050, dilationFactor1051, padding1052, dataFormat1053, conv2d_6NumDims] = ZeroDCE.coder.ops.prepareConvArgs(this.e_conv7_weight, this.e_conv7_bias, this.Vars.ConvStride1031, this.Vars.ConvDilationFactor1032,this.Vars.ConvPadding1033, 1, cat_2NumDims, this.NumDims.e_conv7_weight);
            X1054 = dlarray(single(ZeroDCE.coder.ops.extractIfDlarray(cat_2)));
            Y1055 = dlconv(X1054, weights1048, bias1049, 'Stride', stride1050, 'DilationFactor', dilationFactor1051, 'Padding', padding1052, 'DataFormat', dataFormat1053);
            conv2d_6 = ZeroDCE.coder.ops.extractIfDlarray(Y1055);

            % Set graph output arguments
            conv2d_6NumDims1034 = conv2d_6NumDims;

        end

    end

end