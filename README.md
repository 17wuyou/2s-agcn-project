# 2s-AGCN: PyTorch实现

这是一个基于PyTorch实现的双流自适应图卷积网络（2s-AGCN），用于基于骨骼的动作识别。该项目结构清晰，便于理解和扩展。

## 项目结构

- `agcn/`: 包含所有核心源代码。
  - `model.py`: 定义了模型的所有模块，包括自适应图卷积层和最终的双流网络。
  - `dataset.py`: 包含用于加载骨骼数据的PyTorch Dataset类（当前为演示版本）。
  - `engine.py`: 包含了训练一个周期（epoch）和评估模型的标准函数。
  - `utils.py`: 包含辅助函数，如加载配置文件和生成邻接矩阵。
- `config.yaml`: 存储所有超参数和配置，便于调整实验。
- `requirements.txt`: 项目所需的Python库。
- `run.py`: 训练和评估模型的主执行脚本。

## 使用步骤

### 1. 克隆项目

```bash
git clone <your-repo-link>
cd 2s-agcn-project
```

### 2. 安装依赖

建议在虚拟环境中使用：
```bash
pip install -r requirements.txt
```

### 3. 配置实验

修改 `config.yaml` 文件，设置你的数据路径、模型参数和训练超参数。

**重要提示:** `agcn/utils.py`中的`get_adj_matrix`函数当前是一个**演示版本**。你需要根据你的数据集（如NTU-RGBD）的骨架定义，实现真实的物理邻接矩阵（自身、向心、离心）的生成逻辑。

### 4. 开始训练

```bash
python run.py
```
训练日志将被打印到控制台，最佳模型将根据配置保存在指定路径。