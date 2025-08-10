# 2s-AGCN-PyTorch: 一个清晰易用的实现

这是一个基于PyTorch实现的 **双流自适应图卷积网络（2s-AGCN）**，用于基于骨骼的动作识别。本项目旨在提供一个结构清晰、模块化、易于理解和扩展的代码库，让研究人员和开发者能够快速上手。

**核心特性:**
- **完整的模型实现**: 精准复现了论文中的核心模块，包括自适应图卷积层 (`A_k+B_k+C_k`) 和双流架构。
- **模块化设计**: 代码被清晰地划分为模型、数据加载器、训练引擎和工具函数，便于维护和二次开发。
- **配置驱动**: 所有超参数和路径都通过 `config.yaml` 文件管理，无需修改代码即可进行实验。
- **断点续传**: 内置检查点机制，支持从上次中断处恢复训练，保障长时间任务的稳定性。
- **多阶段训练指南**: 提供从**模拟数据**、**半真实数据**到**真实数据集**的完整、详细的训练流程，确保用户可以循序渐进地运行和验证项目。

## 项目结构

```
2s-agcn-project/
├── agcn/                            # 核心源代码包
│   ├── __init__.py
│   ├── model.py                     # 模型定义 (nn.Module)
│   ├── dataset.py                   # 数据加载器 (Dataset)
│   ├── dataset_simulation.py        # 模拟数据加载器 (Dataset)
│   ├── engine.py                    # 训练/评估循环
│   └── utils.py                     # 辅助函数 (配置加载, 邻接矩阵生成)
├── config.yaml                      # 全局配置文件
├── requirements.txt                 # Python依赖库
├── run.py                           # 主执行脚本
├── generate_dataset.py              # 半真实数据集生成脚本
└── README.md                        # 本说明文件```

---

## 快速开始：环境配置

1.  **克隆项目**
    ```bash
    git clone https://github.com/17wuyou/2s-agcn-project.git
    cd 2s-agcn-project
```

2.  **创建并激活虚拟环境 (推荐)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Windows: venv\Scripts\activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

---

## 训练流程指南

本项目提供了三种不同层次的训练方法，请根据您的需求选择：

### 阶段一：模拟数据训练 (代码验证)

此阶段**无需任何数据文件**，用于快速验证整个代码框架（模型、训练循环、保存/加载逻辑）是否能正常工作。

**步骤:**

1.  **确认`agcn/dataset.py`处于模拟模式**:
    将`agcn/dataset_simulation.py`重命名为 `agcn/dataset.py` （原来的`agcn/dataset.py`需要删除或者重命名）来确保使用的是动态生成随机数据的版本。代码中应该有类似 `"Initializing SkeletonDataset in '{mode}' mode with MOCK data."` 的打印信息。

2.  **运行主程序**:
    ```bash
    python run.py
    ```

**预期结果:**
- 程序会开始训练，但验证集准确率（Val Acc）将在0%附近随机波动。
- `checkpoints` 文件夹会被创建，并生成 `latest_checkpoint.pth` 和 `best_model.pth`。
- 这证明了整个框架是通的，您可以进行下一步了。

---

### 阶段二：半真实数据训练 (模型学习能力验证)

此阶段将生成一个具有类别结构的大型数据集文件，用于验证模型是否具备从数据中学习模式的真实能力。

**步骤:**

1.  **更新配置文件 `config.yaml`**:
    设置数据文件的保存路径。
    ```yaml
    data:
      data_path: "./generated_data" 
      num_joints: 25
      num_classes: 60
    ```

2.  **生成数据集文件**:
    运行 `generate_dataset.py` 脚本。
    ```bash
    python generate_dataset.py
    ```
    执行完毕后，会在项目根目录下创建一个 `generated_data` 文件夹，内含 `train_data.npy`, `train_label.pkl` 等文件。

3.  **切换 `agcn/dataset.py` 至真实数据加载模式**:
    将 `agcn/dataset.py` 的内容替换为**能够加载 `.npy` 和 `.pkl` 文件**的版本。该版本应包含 `np.load()` 和 `pickle.load()` 等函数。

4.  **清理旧的检查点 (重要)**:
    为了避免从模拟数据的状态恢复，请**删除** `checkpoints` 文件夹。
    ```bash
    rm -rf checkpoints
    ```

5.  **开始训练**:
    ```bash
    python run.py
    ```

**预期结果:**
- 程序会加载 `generated_data` 文件夹中的数据进行训练。
- 您应该能观察到**验证集准确率（Val Acc）有显著且持续的提升**，最终可能达到一个较高的水平。
- 这证明了您的模型实现是正确的，并且具备强大的学习能力。

---

### 阶段三：真实数据集训练 (NTU-RGB+D 60)

这是最终阶段，我们将在学术界标准数据集上进行训练，以复现或超越论文报告的性能。

**步骤:**

1.  **获取并预处理NTU-RGB+D数据集**:
    - **申请数据**: 前往 [NTU-RGB+D官网](https://rose1.ntu.edu.sg/dataset/actionRecognition/) 申请并下载 `NTU RGB+D 60 Skeletons` 数据。
    - **使用ST-GCN工具进行预处理**: 这是最关键的一步。请参照ST-GCN官方项目 [https://github.com/yysijie/st-gcn](https://github.com/yysijie/st-gcn) 的指南，运行其 `tools/ntu_gendata/gendata.py` 脚本，生成 **"cross-subject (xsub)"** 划分的数据。
    - **获取产物**: 从 `st-gcn` 的输出目录中，找到 `ntu60_xsub` 文件夹，里面包含了 `train_data.npy`, `train_label.pkl`, `val_data.npy`, `val_label.pkl` 四个文件。

2.  **将数据放入本项目**:
    - 在您的 `2s-agcn-project` 根目录下创建路径 `data/ntu60_xsub`。
    - 将上述四个预处理好的文件拷贝到这个新创建的目录中。

3.  **更新配置文件 `config.yaml`**:
    修改配置以匹配真实数据。
    ```yaml
    # config.yaml
    data:
      data_path: "./data/ntu60_xsub"  # 指向真实数据路径
      num_joints: 25
      num_classes: 60
    
    training:
      epochs: 80
      batch_size: 32  # 如果GPU显存不足，可适当减小
      # 真实数据训练可能需要调整学习率和优化器策略
      learning_rate: 0.1 
      save_path: "./checkpoints/ntu60_xsub_best.pth"
    ```

4.  **更新邻接矩阵生成逻辑 (至关重要)**:
    - 确保 `agcn/utils.py` 中的 `get_adj_matrix` 函数使用的是**为NTU-RGB+D 25个关节点专门设计的版本**，而不是演示版本。它应该能正确生成代表自身、向心、离心连接的三个矩阵。

5.  **清理旧的检查点**:
    再次**删除** `checkpoints` 文件夹，以确保训练从零开始。
    ```bash
    rm -rf checkpoints
    ```

6.  **开始最终训练**:
    ```bash
    python run.py
    ```

**预期结果:**
- 训练过程会比之前长很多。
- 您将看到模型在真实的、复杂的数据上学习，验证集准确率会稳步提升。一个良好配置的训练最终应能达到 **88%** 以上的准确率，与顶级研究成果相媲美。

---

## 断点续传功能

本项目内置了断点续传。如果训练意外中断，只需重新运行 `python run.py`，程序会自动从 `./checkpoints/latest_checkpoint.pth` 文件中加载上次的状态（包括模型权重、优化器状态和epoch编号），并从中断处继续训练。