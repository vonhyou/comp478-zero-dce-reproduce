classdef ConvLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                'e_conv5_bias'
                'e_conv5_weight'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = ZeroDCE.coder.ConvLayer1004(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ZeroDCE.ConvLayer1004(cgInstance.Name);
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
            this_ml.e_conv5_bias = cgInstance.e_conv5_bias;
            this_ml.e_conv5_weight = cgInstance.e_conv5_weight;
        end
    end

    methods
        function this = ConvLayer1004(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'conv2d_4'};
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
            this.e_conv5_bias = mlInstance.e_conv5_bias;
            this.e_conv5_weight = mlInstance.e_conv5_weight;
        end

        function [conv2d_4] = predict(this, cat1000__)
            if isdlarray(cat1000__)
                cat1000_ = stripdims(cat1000__);
            else
                cat1000_ = cat1000__;
            end
            cat1000NumDims = 4;
            cat1000 = ZeroDCE.coder.ops.permuteInputVar(cat1000_, [4 3 1 2], 4);

            [conv2d_4__, conv2d_4NumDims__] = ConvGraph1020(this, cat1000, cat1000NumDims, false);
            conv2d_4_ = ZeroDCE.coder.ops.permuteOutputVar(conv2d_4__, [3 4 2 1], 4);

            conv2d_4 = dlarray(single(conv2d_4_), 'SSCB');
        end

        function [conv2d_4, conv2d_4NumDims1024] = ConvGraph1020(this, cat1000, cat1000NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights1032, bias1033, stride1034, dilationFactor1035, padding1036, dataFormat1037, conv2d_4NumDims] = ZeroDCE.coder.ops.prepareConvArgs(this.e_conv5_weight, this.e_conv5_bias, this.Vars.ConvStride1021, this.Vars.ConvDilationFactor1022,this.Vars.ConvPadding1023, 1, cat1000NumDims, this.NumDims.e_conv5_weight);
            X1038 = dlarray(single(ZeroDCE.coder.ops.extractIfDlarray(cat1000)));
            Y1039 = dlconv(X1038, weights1032, bias1033, 'Stride', stride1034, 'DilationFactor', dilationFactor1035, 'Padding', padding1036, 'DataFormat', dataFormat1037);
            conv2d_4 = ZeroDCE.coder.ops.extractIfDlarray(Y1039);

            % Set graph output arguments
            conv2d_4NumDims1024 = conv2d_4NumDims;

        end

    end

end