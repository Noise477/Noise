import torch
from train import train_model
from evaluate import evaluate_model
from data.dataset import load_data

if __name__ == "__main__":
    # 检查并设置设备为 GPU 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")
    
    # 训练模型，并将模型移到 GPU 或 CPU
    model = train_model(num_epochs=5, device=device)
    
    # 加载测试数据
    _, testloader = load_data()
    
    # 在设备上评估模型
    evaluate_model(model, testloader, device=device)
