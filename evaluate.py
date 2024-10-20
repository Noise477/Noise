import torch

def evaluate_model(model, testloader, device='cpu'):
    # 确保模型在设备上
    model.to(device)
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    
    # 禁用梯度计算，以加速推理
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到设备
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'测试集准确率: {100 * correct / total:.2f}%')
