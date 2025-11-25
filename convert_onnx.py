import torch
import torch.nn as nn
import torch.onnx
import onnx
import model
import os
import sys

class CleanDCE(nn.Module):
    def __init__(self, original_net):
        super(CleanDCE, self).__init__()
        self.e_conv1 = original_net.e_conv1
        self.e_conv2 = original_net.e_conv2
        self.e_conv3 = original_net.e_conv3
        self.e_conv4 = original_net.e_conv4
        self.e_conv5 = original_net.e_conv5
        self.e_conv6 = original_net.e_conv6
        self.e_conv7 = original_net.e_conv7
        self.relu = original_net.relu

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return x_r

def export_clean():
    print("--- 1. Loading Original Model ---")
    pth_path = './ZeroDCEOriginal/Zero-DCE_code/snapshots/Epoch99.pth'
    if not os.path.exists(pth_path):
        print(f"Error: {pth_path} not found.")
        return

    original_net = model.enhance_net_nopool().cuda()
    original_net.load_state_dict(torch.load(pth_path))
    original_net.eval()

    clean_net = CleanDCE(original_net).cuda()
    clean_net.eval()

    # Define paths
    dummy_input = torch.randn(1, 3, 512, 512).cuda()
    temp_onnx = "temp.onnx"
    final_onnx = "ZeroDCE.onnx"

    print("--- 2. Exporting (Opset 13) ---")
    try:
        # Use opset_version=13 if possible (Supported by MATLAB)
        torch.onnx.export(clean_net, 
                          dummy_input, 
                          temp_onnx, 
                          verbose=False,
                          input_names=['input_image'], 
                          output_names=['curve_params'], 
                          opset_version=13, 
                          dynamic_axes={'input_image': {0: 'batch', 2: 'h', 3: 'w'},
                                        'curve_params': {0: 'batch', 2: 'h', 3: 'w'}})
    except Exception as e:
        print(f"\nFATAL ERROR during export: {e}")
        sys.exit(1)

    print("--- 3. Merging and Setting IR Version ---")
    try:
        onnx_model = onnx.load(temp_onnx)
        
        # Manually force IR version 9 (Max supported by my MATLAB ver)
        if onnx_model.ir_version > 9:
            onnx_model.ir_version = 9
            
        onnx.save_model(onnx_model, final_onnx)
        print(f"Success! Model saved as: {final_onnx}")
        
    except Exception as e:
        print(f"Error during merge/save: {e}")
    finally:
        # Clean up temp files
        if os.path.exists(temp_onnx): os.remove(temp_onnx)
        if os.path.exists(temp_onnx + ".data"): os.remove(temp_onnx + ".data")

if __name__ == '__main__':
    export_clean()
