import time  # 引入时间模块
import torch
from models.simple_net import SimpleNet
from data.dataset import load_data
import torch.optim as optim
import torch.nn as nn

def train_model(num_epochs=5, batch_size=32, device='cpu'):
    # 加载数据
    trainloader, _ = load_data(batch_size)
    
    # 创建并移动模型到设备
    model = SimpleNet().to(device)
    print(f"模型所在设备: {next(model.parameters()).device}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 开始训练
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录开始时间
        
        running_loss = 0.0
        for inputs, labels in trainloader:
            # 将输入和标签移动到设备
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新权重
            optimizer.step()
            
            running_loss += loss.item()

        end_time = time.time()  # 记录结束时间
        epoch_time = end_time - start_time  # 计算时间差
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Time: {epoch_time:.2f}秒')

    return model  # 返回训练后的模型
