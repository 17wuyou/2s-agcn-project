import numpy as np
import pickle
import os

def generate_structured_dataset(config):
    """
    生成一个具有结构性的、文件化的“半真实”数据集。
    """
    print("Starting structured dataset generation...")

    # 从配置中获取参数
    data_path = config['data']['data_path']
    num_joints = config['data']['num_joints']
    num_classes = config['data']['num_classes']
    in_channels = config['model']['in_channels']
    num_frames = 300 # 固定帧数

    # 确保保存目录存在
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created data directory: {data_path}")

    # --- 生成训练数据 ---
    num_train_samples = 5000 # 生成5000个训练样本
    train_data = np.zeros((num_train_samples, in_channels, num_frames, num_joints), dtype=np.float32)
    train_labels = np.zeros(num_train_samples, dtype=np.int64)
    train_sample_names = []

    print(f"Generating {num_train_samples} training samples...")
    for i in range(num_train_samples):
        # 确定样本的类别
        label = i % num_classes
        train_labels[i] = label
        train_sample_names.append(f'train_sample_{i+1}')

        # 为每个类别创建一个独特的“原型”姿势
        # 原型 = (类别号 * 偏移量) + 基础噪声
        prototype_pose = np.random.randn(in_channels, num_joints) * 0.1 + (label / num_classes * 2)

        # 围绕原型姿势生成一个动作序列，加入时间和空间上的随机扰动
        for t in range(num_frames):
            # 随时间缓慢变化的噪声 + 姿势噪声
            time_noise = np.sin(2 * np.pi * t / num_frames) * 0.3
            pose_noise = np.random.randn(in_channels, num_joints) * 0.05
            train_data[i, :, t, :] = prototype_pose + time_noise + pose_noise

    # 保存训练数据
    np.save(os.path.join(data_path, 'train_data.npy'), train_data)
    with open(os.path.join(data_path, 'train_label.pkl'), 'wb') as f:
        pickle.dump((train_sample_names, list(train_labels)), f)
    print("Training data saved.")

    # --- 生成验证数据 ---
    num_val_samples = 800 # 生成800个验证样本
    val_data = np.zeros((num_val_samples, in_channels, num_frames, num_joints), dtype=np.float32)
    val_labels = np.zeros(num_val_samples, dtype=np.int64)
    val_sample_names = []

    print(f"Generating {num_val_samples} validation samples...")
    for i in range(num_val_samples):
        label = i % num_classes
        val_labels[i] = label
        val_sample_names.append(f'val_sample_{i+1}')
        
        prototype_pose = np.random.randn(in_channels, num_joints) * 0.1 + (label / num_classes * 2)
        for t in range(num_frames):
            time_noise = np.sin(2 * np.pi * t / num_frames) * 0.3
            pose_noise = np.random.randn(in_channels, num_joints) * 0.05
            val_data[i, :, t, :] = prototype_pose + time_noise + pose_noise

    # 保存验证数据
    np.save(os.path.join(data_path, 'val_data.npy'), val_data)
    with open(os.path.join(data_path, 'val_label.pkl'), 'wb') as f:
        pickle.dump((val_sample_names, list(val_labels)), f)
    print("Validation data saved.")
    
    print("Dataset generation complete!")

if __name__ == '__main__':
    # 从主项目的config.yaml中加载配置来生成数据
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    generate_structured_dataset(config)