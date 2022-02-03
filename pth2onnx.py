import torch
import torch.onnx
from models import Net


def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = Net()  # 导入模型
    model.load_state_dict(torch.load(checkpoint))  # 初始化权重
    model.eval()

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    checkpoint = './models/model.pth'
    onnx_path = './models/model.onnx'
    input = torch.randn(1, 1, 28, 28)
    pth_to_onnx(input, checkpoint, onnx_path)

