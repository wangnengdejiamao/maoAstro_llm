#!/bin/bash
# 白矮星专用模型训练脚本（不使用 Kimi API 版本）
# 使用预定义的模板数据

set -e

echo "========================================"
echo "🌟 Qwen-8B 白矮星专用模型训练"
echo "（备用版本 - 无需 Kimi API）"
echo "========================================"

# 配置
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
PROJECT_NAME="qwen8b-whitewarf"
DATA_DIR="./data/white_dwarf_papers"
MODEL_DIR="./models"

# 创建目录
mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR

echo ""
echo "[1/5] 检查依赖..."
python -c "import json; import torch; print('✓ 依赖检查通过')" 2>/dev/null || {
    echo "安装依赖..."
    pip install torch transformers -q
}

echo ""
echo "[2/5] 生成白矮星训练数据（模板模式）..."
echo "========================================"

# 使用 Python 生成模板训练数据
python3 << 'PYTHON_SCRIPT'
import json
import os

output_dir = "./data/white_dwarf_papers"
os.makedirs(output_dir, exist_ok=True)

# 白矮星领域训练数据（模板生成）
training_data = [
    # 单白矮星 - 基本概念
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "什么是白矮星？"},
            {"role": "assistant", "content": "白矮星是恒星演化的最终阶段之一，是中小质量恒星（< 8 M☉）耗尽核燃料后留下的致密残骸。\n\n主要特征：\n• 质量：~0.6 M☉（典型值），最大 ~1.4 M☉（钱德拉塞卡极限）\n• 半径：~0.01 R☉（地球大小）\n• 密度：~10^6 g/cm³\n• 组成：主要由碳和氧组成（C/O 白矮星），大质量的可能是氧氖镁（O/Ne/Mg），小质量的可能是氦\n\n白矮星不再进行核聚变，通过残余热量辐射冷却，逐渐变暗。典型的冷却时标为数十亿年。"}
        ],
        "metadata": {"topic": "单白矮星", "type": "concept"}
    },
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "什么是钱德拉塞卡极限？"},
            {"role": "assistant", "content": "钱德拉塞卡极限（Chandrasekhar Limit）是白矮星能够通过电子简并压支撑的最大质量，数值约为 1.38-1.4 倍太阳质量（M☉）。\n\n物理意义：\n• 由 S. Chandrasekhar 于 1931 年导出\n• 超过此极限，电子简并压无法抵抗引力\n• 恒星将坍缩为中子星或发生 Ia 型超新星爆发\n\n公式：\nM_ch ≈ 1.38 (Y_e)^2 M☉\n其中 Y_e 是电子平均分子量（对于 C/O 白矮星，Y_e ≈ 0.5）\n\n观测意义：\n• 解释了白矮星质量分布的上限\n• Ia 型超新星前身星质量的标准 candle\n• 验证量子统计力学在天体物理中的应用"}
        ],
        "metadata": {"topic": "单白矮星", "type": "concept"}
    },
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "白矮星的光谱分类有哪些？"},
            {"role": "assistant", "content": "白矮星的光谱分类基于大气层的主要成分：\n\n| 类型 | 特征 | 说明 |\n|------|------|------|\n| **DA** | 氢线 (Hα) | 最普遍 (~80%)，大气层含氢 |\n| **DB** | 氦线 (He I) | ~16%，无氢，温度 11000-30000 K |\n| **DC** | 连续谱，无谱线 | 温度较低或高重力展宽 |\n| **DQ** | 碳线 (C2, C I) | 大气层含碳 |\n| **DZ** | 金属线 (Ca, Mg, Fe) | 含重元素，污染白矮星 |\n| **DAZ** | 氢线 + 金属线 | 含氢且有金属污染 |\n\n后续字母表示额外特征：\n• P = 具有可变性\n• H = 磁场 > 10^6 G（磁性白矮星）\n• X = 具有尘埃盘\n\n示例：DAH = 含氢磁性白矮星"}
        ],
        "metadata": {"topic": "单白矮星", "type": "classification"}
    },
    
    # 双白矮星
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "双白矮星系统如何产生引力波？"},
            {"role": "assistant", "content": "双白矮星系统通过引力波辐射损失轨道能量，导致轨道收缩，这是广义相对论的重要预言。\n\n物理机制：\n• 两个致密天体绕转产生时变四极矩\n• 辐射引力波带走能量和角动量\n• 轨道半径减小，周期变短\n\n引力波特征：\n• 频率：mHz 范围（0.1-100 mHz）\n• 振幅：h ~ 10^-23 到 10^-20\n• 是空间引力波探测器（LISA、天琴、太极）的主要目标\n\n并合时标：\nτ_merge ≈ 10^7 (P/1hr)^(8/3) (M/0.6M☉)^(-5/3) 年\n\n科学价值：\n• 检验广义相对论在强场极限\n• 测定白矮星质量分布\n• 研究双星演化终点"}
        ],
        "metadata": {"topic": "双白矮星", "type": "concept"}
    },
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "双白矮星并合一定会产生超新星吗？"},
            {"role": "assistant", "content": "**不是**。双白矮星并合不一定会产生超新星，取决于总质量。\n\n两种可能结果：\n\n1. **超钱德拉塞卡并合** (M_total > 1.4 M☉)：\n   • 可能产生 Ia 型超新星（双简并通道）\n   • 或形成 R CrB 型星（如果保留氦包层）\n   • 或坍缩为中子星\n\n2. **亚钱德拉塞卡并合** (M_total < 1.4 M☉)：\n   • 形成更大质量的白矮星\n   • 可能成为 AM CVn 型系统\n   • 或形成 R CrB 型星\n   • **不会**产生超新星\n\n观测证据：\n• 大多数双白矮星总质量 < 1.4 M☉\n• 银河系双白矮星并合率 ~10^-13 /yr\n• Ia 型超新星需要其他触发机制（如氦吸积点燃）\n\n因此，\"双白矮星并合总是产生超新星\"是**错误**的说法。"}
        ],
        "metadata": {"topic": "双白矮星", "type": "anti_hallucination"}
    },
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "什么是双白矮星并合的产物？"},
            {"role": "assistant", "content": "双白矮星并合的产物取决于总质量和组成：\n\n| 总质量 | 可能产物 |\n|--------|----------|\n| < 1.0 M☉ | 更大质量白矮星，或 AM CVn 前身 |\n| 1.0-1.4 M☉ | R CrB 型星，或极端氦星 |\n| > 1.4 M☉ | Ia 型超新星，或中子星 |\n\n具体过程：\n\n1. **潮汐破坏**：低质量白矮星被潮汐撕裂\n2. **吸积盘形成**：物质形成围绕主星的盘\n3. **热核反应**：\n   - 亚钱德拉：稳定燃烧或 R CrB 型星\n   - 超钱德拉：失控碳燃烧 → 超新星\n\n关键参数：\n• 质量比 q = M2/M1\n• 总质量 M_total\n• 组成（C/O vs He）\n\n观测候选：\n• V1309 Sco：并合事件\n• R CrB 型星：可能的并合产物"}
        ],
        "metadata": {"topic": "双白矮星", "type": "concept"}
    },
    
    # 磁性白矮星
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "磁性白矮星的磁场是怎么产生的？"},
            {"role": "assistant", "content": "磁性白矮星（MWD）的强磁场（10^3 - 10^9 G）可能来源于多种机制，目前尚无定论：\n\n可能机制：\n\n1. **原恒星磁场冻结**\n   • 主序星磁场在坍缩过程中放大\n   • B ∝ R^-2，坍缩后磁场增强 10^4-10^8 倍\n   • 可解释 10^6-10^8 G 的磁场\n\n2. **双星并合发电机**\n   • 双白矮星并合时的发电机效应\n   • 湍流和差动旋转产生磁场\n   • 可解释极高场（>10^8 G）\n\n3. **氦核闪期间发电机**\n   • 早期演化阶段的湍流发电机\n   • 对流区产生磁场并冻结\n\n观测统计：\n• 约 10-20% 的白矮星有可探测磁场\n• 磁场分布双峰：低场（<10^6 G）和高场（>10^7 G）\n• 极高场星（>10^9 G）：如 GRW +70 8247\n\n研究意义：\n• 验证发电机理论\n• 了解白矮星形成历史\n• 磁缀食双星的物理"}
        ],
        "metadata": {"topic": "磁性白矮星", "type": "concept"}
    },
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "所有白矮星都有强磁场吗？"},
            {"role": "assistant", "content": "**不是**。只有约 **10-20%** 的白矮星表现出可探测的磁场。\n\n统计数据：\n• 磁性白矮星比例：~10-20%\n• 低场（<10^6 G）：难以检测\n• 中强场（10^6-10^8 G）：可通过塞曼分裂检测\n• 高场（>10^8 G）：强烈偏振和谱线分裂\n\n磁场分布：\n• 双峰分布暗示两种起源机制\n• 高场星可能来自并合\n• 低场星可能来自原恒星磁场\n\n因此，\"所有白矮星都有强磁场\"是**错误**的说法。大多数白矮星磁场很弱或没有可探测磁场。"}
        ],
        "metadata": {"topic": "磁性白矮星", "type": "anti_hallucination"}
    },
    
    # 吸积白矮星
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "什么是激变变星（Cataclysmic Variable）？"},
            {"role": "assistant", "content": "激变变星（Cataclysmic Variables, CVs）是由白矮星从伴星吸积物质的双星系统，表现出剧烈的光度变化。\n\n系统组成：\n• **白矮星**：主星，吸积体\n• **供体星**：通常为红矮星（主序星）\n• **吸积盘**：围绕白矮星的气态盘\n• **热点**：吸积流冲击盘面的区域\n\n主要类型：\n\n| 类型 | 特征 | 周期 |\n|------|------|------|\n| **矮新星** (DN) | 周期性爆发（2-15 mag）| 数天到数月 |\n| **经典新星** (CN) | 一次性大爆发（10-20 mag）| 数十年到数千年 |\n| **再发新星** (RN) | 多次复发 | 10-100 年 |\n| **极星** (AM Her) | 强磁场，无盘，自转同步 | 1-4 小时 |\n| **中介极星** (DQ Her) | 中强磁场，部分盘 | 2-4 小时 |\n| **类新星** (NL) | 高态，无爆发 | - |\n| **磁星** (IP) | X射线亮，吸积柱 | - |\n\n光变机制：\n• 矮新星：吸积盘不稳定性（DIM）\n• 新星：表面积累氢的热核爆发\n• 极星：吸积流直接撞击磁极"}
        ],
        "metadata": {"topic": "吸积白矮星", "type": "concept"}
    },
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "什么是 AM CVn 系统？"},
            {"role": "assistant", "content": "AM CVn 系统是一类**超紧凑双白矮星系统**，是已知轨道周期最短的双星系统。\n\n系统组成：\n• **主星**：白矮星（通常为 C/O）\n• **供体星**：氦白矮星或裸露氦核（已失去氢包层）\n\n⚠️ **重要**：AM CVn **绝对不含中子星**！这是最常见的误解。\n\n关键特征：\n• **周期**：5-65 分钟（最短周期双星）\n• **距离**：通常 < 1000 pc\n• **光变**：吸积盘不稳定性导致\n• **光谱**：氦线主导，无氢线\n\n物理机制：\n1. **质量转移**：氦通过内拉格朗日点转移\n2. **吸积盘**：形成围绕主星的盘\n3. **光变**：\n   - 高态/低态转换（DIM 模型）\n   - 超爆发（superoutburst）\n\n科学意义：\n• **引力波源**：LISA 重要目标\n• **Ia 型超新星前身**：双简并通道\n• **极端物理**：验证吸积盘理论\n\n代表天体：\n• **AM CVn**：原型，周期 17.1 分钟\n• **HM Cnc**：周期 5.4 分钟（最短）\n• **V407 Vul**：X射线亮"}
        ],
        "metadata": {"topic": "吸积白矮星", "type": "concept"}
    },
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "AM CVn 系统包含中子星吗？"},
            {"role": "assistant", "content": "**绝对不含中子星！**\n\nAM CVn 系统是**双白矮星系统**，由以下组成：\n• 一颗白矮星（主星，吸积体）\n• 一颗氦白矮星或裸露氦核（供体星）\n\n为什么不是中子星：\n• 观测质量函数与白矮星一致\n• 光谱显示氦线，无中子星特征\n• 周期 5-65 分钟与双白矮星并合演化模型一致\n• X射线特性与吸积白矮星一致\n\n常见误解来源：\n• 周期极短，与某些中子星双星类似\n• 高能量辐射，误认为脉冲星\n\n正确理解：\nAM CVn 名称来源于原型星 AM Canum Venaticorum，是**双白矮星吸积系统**的原型，与 AM Herculis（极星，也是白矮星系统）不同。两者都不含中子星！"}
        ],
        "metadata": {"topic": "吸积白矮星", "type": "anti_hallucination"}
    },
    {
        "messages": [
            {"role": "system", "content": "你是白矮星天体物理学专家。"},
            {"role": "user", "content": "吸积白矮星的光变是由什么引起的？"},
            {"role": "assistant", "content": "吸积白矮星的光变机制因系统类型而异：\n\n1. **矮新星（Dwarf Novae）**：\n   • 机制：**吸积盘不稳定性模型（DIM）**\n   • 低态：冷盘，低吸积率\n   • 高态：热盘， ionization front 传播\n   • 周期：数天到数月\n\n2. **经典新星（Classical Novae）**：\n   • 机制：**表面热核爆发**\n   • 积累氢层达到临界温度和密度\n   • 失控 CNO 循环\n   • 亮度增加 10^4-10^6 倍\n\n3. **极星（AM Her 型）**：\n   • 机制：**吸积流冲击磁极**\n   • 无吸积盘，物质沿磁力线\n   • 软 X射线 + 光学/UV 辐射\n   • 自转与轨道同步\n\n4. **中介极星（DQ Her 型）**：\n   • 机制：**部分盘 + 磁吸积**\n   • 内盘被磁场截断\n   • 吸积柱产生 X射线\n\n❌ **错误观点**：光变是由"潮汐相互作用"引起的。\n潮汐力影响轨道演化，不直接导致周期性光变。"}
        ],
        "metadata": {"topic": "吸积白矮星", "type": "anti_hallucination"}
    },
]

# 添加更多变体
import random

def create_variants(base_data, n_variants=3):
    """创建数据的变体"""
    variants = []
    for item in base_data:
        variants.append(item)
        # 这里可以添加数据增强逻辑
    return variants

# 保存训练数据
output_file = os.path.join(output_dir, "wd_training_data.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"✓ 生成 {len(training_data)} 条训练数据")
print(f"✓ 保存到: {output_file}")

# 同时生成评估数据（MCQ 格式）
eval_data = [
    {
        "id": "wd_001",
        "question": "钱德拉塞卡极限的数值约为：",
        "options": {"A": "0.5 M☉", "B": "1.0 M☉", "C": "1.4 M☉", "D": "2.0 M☉"},
        "correct": "C",
        "subfield": "single_wd",
        "difficulty": "easy"
    },
    {
        "id": "wd_002",
        "question": "双白矮星并合一定会产生超新星吗？",
        "options": {
            "A": "是的，所有并合都会产生", 
            "B": "只有总质量超过钱德拉塞卡极限才可能产生",
            "C": "永远不会产生",
            "D": "取决于轨道倾角"
        },
        "correct": "B",
        "subfield": "double_wd",
        "difficulty": "medium"
    },
    {
        "id": "wd_003",
        "question": "AM CVn 系统包含什么？",
        "options": {
            "A": "白矮星 + 中子星",
            "B": "两颗白矮星",
            "C": "白矮星 + 主序星",
            "D": "中子星 + 红巨星"
        },
        "correct": "B",
        "subfield": "accreting_wd",
        "difficulty": "hard"
    },
]

eval_file = os.path.join(output_dir, "wd_eval.json")
with open(eval_file, 'w', encoding='utf-8') as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print(f"✓ 生成 {len(eval_data)} 条评估数据")
print(f"✓ 保存到: {eval_file}")
PYTHON_SCRIPT

echo ""
echo "[3/5] 检查 LLaMA-Factory..."
if ! command -v llamafactory-cli &> /dev/null; then
    echo "安装 LLaMA-Factory..."
    pip install "llamafactory[torch,metrics]" -q
fi
echo "✓ LLaMA-Factory 已就绪"

echo ""
echo "[4/5] 开始模型训练..."
echo "========================================"

# Stage 2: 监督微调
llamafactory-cli train \
    --stage sft \
    --model_name_or_path $BASE_MODEL \
    --do_train \
    --dataset wd_training_data \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --output_dir $MODEL_DIR/${PROJECT_NAME}-sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 4e-5 \
    --num_train_epochs 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 \
    --plot_loss \
    --overwrite_output_dir

echo ""
echo ">>> 合并权重"

llamafactory-cli export \
    --model_name_or_path $BASE_MODEL \
    --adapter_name_or_path $MODEL_DIR/${PROJECT_NAME}-sft \
    --finetuning_type lora \
    --template qwen \
    --export_dir $MODEL_DIR/${PROJECT_NAME}-final \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format false

echo ""
echo "[5/5] 评估模型..."
echo "========================================"

python wd_evaluator.py \
    --model $MODEL_DIR/${PROJECT_NAME}-final \
    --interface hf \
    --output-dir ./eval_results

echo ""
echo "========================================"
echo "✅ 训练完成!"
echo "========================================"
echo "模型位置: $MODEL_DIR/${PROJECT_NAME}-final"
echo ""
echo "使用方法:"
echo "  ollama create whitewarf -f Modelfile.whitewarf"
echo "  ollama run whitewarf"
echo "========================================"
