classdef ConvLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        e_conv1_bias
        e_conv1_weight
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
                'e_conv1_bias'
                'e_conv1_weight'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = ZeroDCE.coder.ConvLayer1000(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ZeroDCE.ConvLayer1000(cgInstance.Name);
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
            this_ml.e_conv1_bias = cgInstance.e_conv1_bias;
            this_ml.e_conv1_weight = cgInstance.e_conv1_weight;
        end
    end

    methods
        function this = ConvLayer1000(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'conv2d'};
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
            this.e_conv1_bias = mlInstance.e_conv1_bias;
            this.e_conv1_weight = mlInstance.e_conv1_weight;
        end

        function [conv2d] = predict(this, input_image__)
            if isdlarray(input_image__)
                input_image_ = stripdims(input_image__);
            else
                input_image_ = input_image__;
            end
            input_imageNumDims = 4;
            input_image = ZeroDCE.coder.ops.permuteInputVar(input_image_, [4 3 1 2], 4);

            [conv2d__, conv2dNumDims__] = ConvGraph1000(this, input_image, input_imageNumDims, false);
            conv2d_ = ZeroDCE.coder.ops.permuteOutputVar(conv2d__, [3 4 2 1], 4);

            conv2d = dlarray(single(conv2d_), 'SSCB');
        end

        function [conv2d, conv2dNumDims1004] = ConvGraph1000(this, input_image, input_imageNumDims, Training)

            % Execute the operators:
            % Conv:
            [weights1000, bias1001, stride1002, dilationFactor1003, padding1004, dataFormat1005, conv2dNumDims] = ZeroDCE.coder.ops.prepareConvArgs(this.e_conv1_weight, this.e_conv1_bias, this.Vars.ConvStride1001, this.Vars.ConvDilationFactor1002,this.Vars.ConvPadding1003, 1, input_imageNumDims, this.NumDims.e_conv1_weight);
            X1006 = dlarray(single(ZeroDCE.coder.ops.extractIfDlarray(input_image)));
            Y1007 = dlconv(X1006, weights1000, bias1001, 'Stride', stride1002, 'DilationFactor', dilationFactor1003, 'Padding', padding1004, 'DataFormat', dataFormat1005);
            conv2d = ZeroDCE.coder.ops.extractIfDlarray(Y1007);

            % Set graph output arguments
            conv2dNumDims1004 = conv2dNumDims;

        end

    end

end