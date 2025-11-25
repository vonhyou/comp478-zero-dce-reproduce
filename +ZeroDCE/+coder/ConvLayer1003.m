classdef ConvLayer1003 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                'e_conv4_bias'
                'e_conv4_weight'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = ZeroDCE.coder.ConvLayer1003(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = ZeroDCE.ConvLayer1003(cgInstance.Name);
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
            this_ml.e_conv4_bias = cgInstance.e_conv4_bias;
            this_ml.e_conv4_weight = cgInstance.e_conv4_weight;
        end
    end

    methods
        function this = ConvLayer1003(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'conv2d_3'};
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
            this.e_conv4_bias = mlInstance.e_conv4_bias;
            this.e_conv4_weight = mlInstance.e_conv4_weight;
        end

        function [conv2d_3] = predict(this, relu_2__)
            if isdlarray(relu_2__)
                relu_2_ = stripdims(relu_2__);
            else
                relu_2_ = relu_2__;
            end
            relu_2NumDims = 4;
            relu_2 = ZeroDCE.coder.ops.permuteInputVar(relu_2_, [4 3 1 2], 4);

            [conv2d_3__, conv2d_3NumDims__] = ConvGraph1015(this, relu_2, relu_2NumDims, false);
            conv2d_3_ = ZeroDCE.coder.ops.permuteOutputVar(conv2d_3__, [3 4 2 1], 4);

            conv2d_3 = dlarray(single(conv2d_3_), 'SSCB');
        end

        function [conv2d_3, conv2d_3NumDims1019] = ConvGraph1015(this, relu_2, relu_2NumDims, Training)

            % Execute the operators:
            % Conv:
            [weights1024, bias1025, stride1026, dilationFactor1027, padding1028, dataFormat1029, conv2d_3NumDims] = ZeroDCE.coder.ops.prepareConvArgs(this.e_conv4_weight, this.e_conv4_bias, this.Vars.ConvStride1016, this.Vars.ConvDilationFactor1017,this.Vars.ConvPadding1018, 1, relu_2NumDims, this.NumDims.e_conv4_weight);
            X1030 = dlarray(single(ZeroDCE.coder.ops.extractIfDlarray(relu_2)));
            Y1031 = dlconv(X1030, weights1024, bias1025, 'Stride', stride1026, 'DilationFactor', dilationFactor1027, 'Padding', padding1028, 'DataFormat', dataFormat1029);
            conv2d_3 = ZeroDCE.coder.ops.extractIfDlarray(Y1031);

            % Set graph output arguments
            conv2d_3NumDims1019 = conv2d_3NumDims;

        end

    end

end