"""
训练 & 推理集中配置文件。
所有路径均为相对于项目根目录的路径，或绝对路径。
"""
import os

# ================= 项目根目录 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 模型 =================
MODEL_PATH = "/home/hank/llama-3.1-8b"

# ================= LoRA =================
LORA_R = 16
LORA_ALPHA = 32         # alpha = 2*r，缩放系数 = alpha/r = 2
LORA_DROPOUT = 0.1
# Qwen3/Llama 的线性层名称，"all-linear" 表示全部
LORA_TARGET_MODULES = "all-linear"

# ================= 数据（相对于项目根目录） =================
DATA_FILES = [
    "/home/hank/PycharmProjects/PythonProject2/LLM1/kaguya.json",
]
TRAIN_VAL_SPLIT = 0.9   # 80% 训练 / 20% 验证
USE_THINKING = False         # 是否在训练数据中包含思考链 (<think> 标签)

# ================= 训练参数 =================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "LLM2", "main", "lora_output")
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM = 2          # 等效 batch_size = 2
LEARNING_RATE = 2e-4    # 15条极小数���集，保守学习率
EPOCHS = 1              # 小数据集多跑几轮
EVAL_STEPS = 3          # 高频评估（约每轮评估2次）
WARMUP_STEPS= 10
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

# ================= 设备 =================
FORCE_MPS = False
MPS_FAST_MATH = True

# ================= 推理 =================
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
MPS_GREEDY_DECODING = True
