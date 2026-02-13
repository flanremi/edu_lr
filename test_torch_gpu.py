# -*- coding: utf-8 -*-
"""测试当前环境 PyTorch 是否可用 GPU（CUDA）。"""
import sys

def main():
    try:
        import torch
    except ImportError:
        print("未安装 PyTorch，请先: pip install torch")
        sys.exit(1)

    print("PyTorch 版本:", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    print("CUDA 可用:", cuda_ok)
    if cuda_ok:
        print("CUDA 版本:", torch.version.cuda)
        print("GPU 数量:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        # 简单计算测试
        x = torch.randn(3, 3).cuda()
        y = x @ x
        print("GPU 计算测试: 通过")
    else:
        print("当前仅使用 CPU，无可用 GPU。")

if __name__ == "__main__":
    main()
