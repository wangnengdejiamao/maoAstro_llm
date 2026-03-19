# 🔭 天文PDF问答数据集生成 - 实时状态

## 🚀 当前状态

- **启动时间**: 2025-03-11
- **API模式**: ✅ 已启用 (Kimi API)
- **处理进度**: 运行中

## 📊 对比：规则 vs API 生成

### 规则生成示例
```
问题: 论文中展示的光变曲线有什么特征？
答案: 光变曲线呈现magnitude diagram特征。观测显示...
```

### API生成示例 ⭐
```
问题: 在 Gaia 色–绝对星等赫罗图上，标准灾变双星（CV）的轨道周期 
      P_orb 如何随位置变化？这一趋势与白矮星主序和主序星的位置关系如何？

答案: Abrahams 等人利用 Gaia DR2/EDR3 发现，70 min ≤ P_orb ≤ 8 hr 的标准 CV 
      在 G_BP–G_RP 色–M_G 赫罗图上呈近似单调序列：P_orb 沿"白矮星主序—主序星" 
      连线大致正交方向递增... P_orb ≈ 2.2 hr 对应 ⟨G_BP–G_RP⟩ ≃ 0.40 mag、
      M_G ≃ 9.5 mag；P_orb ≈ 3.2 hr 对应 ⟨G_BP–G_RP⟩ ≃ 0.65 mag、
      M_G ≃ 8.7 mag...
```

## 🎯 质量提升

| 指标 | 规则生成 | API生成 |
|-----|---------|---------|
| 问题深度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 答案准确性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 专业术语 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 具体数值 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 物理意义解释 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 📝 使用方法

### 查看实时进度
```bash
# 查看已处理数量
ls output/qa_hybrid/cache/*.json | wc -l

# 运行监控脚本
./monitor_progress.sh
```

### 查看最新结果
```bash
# 查看统计
python analyze_qa_results.py

# 查看API生成的高质量问答
python3 -c "
import json
with open('output/qa_hybrid/qa_dataset_full.json') as f:
    qa = [q for q in json.load(f) if q['generation_method']=='api_based']
    for q in qa[:3]:
        print(f'Q: {q[\"question\"][:80]}...')
        print(f'A: {q[\"answer\"][:150]}...')
        print()
"
```

## ⚠️ 注意事项

1. **API速率限制**: Kimi API 可能有速率限制，已添加延迟和重试机制
2. **处理时间**: 使用 API 后处理速度变慢，预计完成全部 259 个文件需要 2-4 小时
3. **多Key轮询**: 5个 API key 会自动轮询使用

## 📁 输出文件

生成完成后，输出目录结构：
```
output/qa_hybrid/
├── qa_dataset_full.json          # 完整数据集 (~2万条)
├── qa_hr_diagram.json            # 赫罗图问题
├── qa_sed.json                   # SED问题
├── qa_light_curve.json           # 光变曲线问题
├── qa_period.json                # 周期问题
├── qa_xray.json                  # X射线问题
├── qa_spectrum.json              # 光谱问题
├── train_conversations.json      # 训练集 (80%)
└── test_conversations.json       # 测试集 (20%)
```

## 🔍 数据特点

1. **多主题覆盖**: 赫罗图、SED、光变曲线、周期、X射线、光谱、灾变变星、双星
2. **来源可追溯**: 每个问答包含源文件名和页码
3. **置信度标记**: 规则生成(0.6-0.8) vs API生成(0.9)
4. **对话格式**: 可直接用于大模型微调

---
*生成任务正在后台运行，可随时查看进度*
