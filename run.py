# run.py (Corrected Version)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os # 1. 导入 os 模块

from agcn.utils import load_config, get_adj_matrix
from agcn.dataset import SkeletonDataset
from agcn.model import Model_2sAGCN
from agcn.engine import train_one_epoch, evaluate

def main():
    # 1. 加载配置
    config = load_config('config.yaml')
    print("Configuration loaded.")

    # 2. 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 准备数据
    train_dataset = SkeletonDataset(config, mode='train')
    val_dataset = SkeletonDataset(config, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    print("Data loaded.")

    # 4. 准备模型
    A = get_adj_matrix(config['data']['num_joints']).to(device)
    bone_pairs = [(i, i + 1) for i in range(config['data']['num_joints'] - 1)] 
    model = Model_2sAGCN(
        num_joints=config['data']['num_joints'],
        num_classes=config['data']['num_classes'],
        A=A,
        bone_pairs=bone_pairs
    ).to(device)
    print("Model created.")

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # --- *** 这是修正的部分 *** ---
    # 6. 准备保存目录
    save_path = config['training']['save_path']
    save_dir = os.path.dirname(save_path) # 获取目录路径 ./checkpoints
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    # --- *** 修正结束 *** ---

    # 7. 训练循环
    best_acc = 0.0
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} Results: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path) # 使用变量 save_path
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

    print("Training finished.")

if __name__ == '__main__':
    main()