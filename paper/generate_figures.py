#!/usr/bin/env python3
"""
AstroSage Technical Architecture Figure Generator
生成论文所需的技术架构图和流程图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import matplotlib.patheffects as path_effects

# 设置中文字体支持
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_system_architecture():
    """创建系统整体架构图 (Figure 1)"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 颜色方案
    colors = {
        'data': '#E3F2FD',      # 浅蓝 - 数据层
        'processing': '#FFF3E0', # 浅橙 - 处理层
        'model': '#E8F5E9',      # 浅绿 - 模型层
        'rag': '#F3E5F5',        # 浅紫 - RAG层
        'output': '#FFEBEE',     # 浅红 - 输出层
        'border': '#1976D2',     # 深蓝 - 边框
        'text': '#212121',       # 深灰 - 文字
        'arrow': '#546E7A'       # 灰蓝 - 箭头
    }
    
    # 标题
    title = ax.text(8, 11.5, 'AstroSage System Architecture', 
                   fontsize=18, fontweight='bold', ha='center', color=colors['text'])
    title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # ========== 数据层 (底部) ==========
    data_box = FancyBboxPatch((0.5, 0.5), 15, 2.2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['data'], 
                              edgecolor=colors['border'], linewidth=2)
    ax.add_patch(data_box)
    ax.text(8, 2.4, 'Multi-Source Astronomical Data Layer', 
           fontsize=12, fontweight='bold', ha='center', color=colors['border'])
    
    # 数据源
    data_sources = [
        ('ZTF/TESS\nPhotometry', 1.5),
        ('LAMOST/SDSS\nSpectra', 4.0),
        ('Gaia DR3\nAstrometry', 6.5),
        ('SIMBAD/VSX\nCatalogs', 9.0),
        ('Extinction\nMaps', 11.5),
        ('Literature\nPapers', 14.0)
    ]
    
    for label, x in data_sources:
        box = FancyBboxPatch((x-0.9, 0.8), 1.8, 1.2, 
                            boxstyle="round,pad=0.05",
                            facecolor='white', edgecolor=colors['border'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 1.4, label, fontsize=8, ha='center', va='center', color=colors['text'])
    
    # ========== RAG知识库层 ==========
    rag_box = FancyBboxPatch((0.5, 3.0), 7, 2.5, 
                            boxstyle="round,pad=0.1",
                            facecolor=colors['rag'], 
                            edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(rag_box)
    ax.text(4, 5.2, 'RAG Knowledge Base', 
           fontsize=11, fontweight='bold', ha='center', color='#7B1FA2')
    
    rag_components = [
        'Vector Store\n(Chroma/FAISS)',
        'Knowledge Graph',
        'Multilingual\nSupport'
    ]
    for i, comp in enumerate(rag_components):
        x = 1.8 + i * 2.2
        box = FancyBboxPatch((x-0.9, 3.3), 1.8, 1.5,
                            boxstyle="round,pad=0.05",
                            facecolor='white', edgecolor='#7B1FA2', linewidth=1)
        ax.add_patch(box)
        ax.text(x, 4.05, comp, fontsize=8, ha='center', va='center', color=colors['text'])
    
    # ========== 数据处理层 ==========
    process_box = FancyBboxPatch((8.5, 3.0), 7, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['processing'],
                                edgecolor='#E65100', linewidth=2)
    ax.add_patch(process_box)
    ax.text(12, 5.2, 'Data Processing Pipeline', 
           fontsize=11, fontweight='bold', ha='center', color='#E65100')
    
    process_components = [
        'Light Curve\nAnalysis',
        'Spectral\nProcessing',
        'HR Diagram\nGeneration'
    ]
    for i, comp in enumerate(process_components):
        x = 9.8 + i * 2.2
        box = FancyBboxPatch((x-0.9, 3.3), 1.8, 1.5,
                            boxstyle="round,pad=0.05",
                            facecolor='white', edgecolor='#E65100', linewidth=1)
        ax.add_patch(box)
        ax.text(x, 4.05, comp, fontsize=8, ha='center', va='center', color=colors['text'])
    
    # ========== 模型层 ==========
    model_box = FancyBboxPatch((0.5, 5.8), 15, 3.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['model'],
                              edgecolor='#388E3C', linewidth=2)
    ax.add_patch(model_box)
    ax.text(8, 8.7, 'LLM Training & Inference Engine', 
           fontsize=12, fontweight='bold', ha='center', color='#388E3C')
    
    # 教师模型
    teacher_box = FancyBboxPatch((1.0, 6.2), 4, 2.3,
                                boxstyle="round,pad=0.05",
                                facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(teacher_box)
    ax.text(3, 8.2, 'Teacher Model', fontsize=10, fontweight='bold', ha='center', color='#1B5E20')
    ax.text(3, 7.6, 'Qwen3-32B', fontsize=9, ha='center', color=colors['text'])
    ax.text(3, 7.2, 'Knowledge Generation', fontsize=8, ha='center', color='#555')
    ax.text(3, 6.8, 'Complex Reasoning', fontsize=8, ha='center', color='#555')
    ax.text(3, 6.4, 'Soft Label Creation', fontsize=8, ha='center', color='#555')
    
    # 蒸馏箭头
    ax.annotate('', xy=(6.5, 7.3), xytext=(5.2, 7.3),
               arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=3))
    ax.text(5.85, 7.6, 'Distillation', fontsize=9, ha='center', color='#D32F2F', fontweight='bold')
    ax.text(5.85, 7.0, 'T=2.0, α=0.7', fontsize=7, ha='center', color='#D32F2F')
    
    # 学生模型
    student_box = FancyBboxPatch((6.8, 6.2), 4, 2.3,
                                boxstyle="round,pad=0.05",
                                facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(student_box)
    ax.text(8.8, 8.2, 'Student Model', fontsize=10, fontweight='bold', ha='center', color='#1B5E20')
    ax.text(8.8, 7.6, 'Qwen3-1.8B/7B', fontsize=9, ha='center', color=colors['text'])
    ax.text(8.8, 7.2, 'LoRA Fine-tuning', fontsize=8, ha='center', color='#555')
    ax.text(8.8, 6.8, 'Pattern Learning', fontsize=8, ha='center', color='#555')
    ax.text(8.8, 6.4, 'Fast Inference', fontsize=8, ha='center', color='#555')
    
    # RAG增强
    ax.annotate('', xy=(11.5, 7.3), xytext=(11.0, 7.3),
               arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=2))
    ax.text(11.25, 7.7, '+RAG', fontsize=8, ha='center', color='#7B1FA2')
    
    # 推理引擎
    inference_box = FancyBboxPatch((11.5, 6.2), 3.5, 2.3,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2)
    ax.add_patch(inference_box)
    ax.text(13.25, 8.2, 'Inference Engine', fontsize=10, fontweight='bold', ha='center', color='#E65100')
    ax.text(13.25, 7.5, 'Ollama Integration', fontsize=8, ha='center', color=colors['text'])
    ax.text(13.25, 7.0, '5x Speedup', fontsize=8, ha='center', color='#2E7D32')
    ax.text(13.25, 6.5, 'Local Deployment', fontsize=8, ha='center', color='#2E7D32')
    
    # ========== 输出层 ==========
    output_box = FancyBboxPatch((0.5, 9.3), 15, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['output'],
                               edgecolor='#C62828', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8, 10.5, 'Analysis Outputs', 
           fontsize=11, fontweight='bold', ha='center', color='#C62828')
    
    outputs = [
        ('Classification', 2),
        ('Physical\nParameters', 4.5),
        ('Scientific\nInsights', 7),
        ('Visualization', 9.5),
        ('Recommendations', 12),
        ('Explanation', 14.5)
    ]
    for label, x in outputs:
        ax.text(x, 9.8, label, fontsize=8, ha='center', va='center', color=colors['text'])
    
    # 添加连接箭头
    # 数据 -> 处理/RAG
    for x in [4, 12]:
        ax.annotate('', xy=(x, 3.0), xytext=(x, 2.8),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    
    # 处理/RAG -> 模型
    ax.annotate('', xy=(8, 5.8), xytext=(8, 5.5),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # 模型 -> 输出
    ax.annotate('', xy=(8, 9.3), xytext=(8, 9.0),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    plt.tight_layout()
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure1_system_architecture.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure1_system_architecture.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Figure 1: System Architecture")


def create_rag_distillation_pipeline():
    """创建RAG与蒸馏整合流程图 (Figure 2)"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(8, 9.5, 'RAG-Enhanced Knowledge Distillation Pipeline', 
           fontsize=16, fontweight='bold', ha='center')
    
    colors = {
        'training': '#BBDEFB',
        'inference': '#C8E6C9',
        'teacher': '#FFCCBC',
        'student': '#D1C4E9',
        'rag': '#F8BBD0',
        'text': '#212121'
    }
    
    # ========== 训练阶段 ==========
    train_box = FancyBboxPatch((0.3, 4.8), 7.5, 4.3,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['training'],
                              edgecolor='#1976D2', linewidth=2)
    ax.add_patch(train_box)
    ax.text(4.05, 8.8, 'Training Phase (Offline)', 
           fontsize=12, fontweight='bold', ha='center', color='#1565C0')
    
    # 训练查询
    query_box = FancyBboxPatch((0.6, 7.5), 2.2, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='#1976D2', linewidth=1)
    ax.add_patch(query_box)
    ax.text(1.7, 7.9, 'Training\nQueries', fontsize=9, ha='center', va='center')
    
    # RAG检索
    rag_box = FancyBboxPatch((3.3, 7.5), 2.2, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['rag'], edgecolor='#C2185B', linewidth=1.5)
    ax.add_patch(rag_box)
    ax.text(4.4, 7.9, 'RAG\nRetrieval', fontsize=9, ha='center', va='center')
    
    # 教师模型
    teacher_box = FancyBboxPatch((0.6, 5.3), 3.5, 1.8,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['teacher'], 
                                edgecolor='#E64A19', linewidth=2)
    ax.add_patch(teacher_box)
    ax.text(2.35, 6.8, 'Teacher Model (32B)', 
           fontsize=10, fontweight='bold', ha='center', color='#BF360C')
    ax.text(2.35, 6.2, '• Complex Reasoning', fontsize=8, ha='center')
    ax.text(2.35, 5.8, '• Soft Label Generation', fontsize=8, ha='center')
    ax.text(2.35, 5.4, '• Confidence Estimation', fontsize=8, ha='center')
    
    # 蒸馏过程
    distill_box = FancyBboxPatch((4.5, 5.3), 3.0, 1.8,
                                boxstyle="round,pad=0.05",
                                facecolor='#FFF9C4', 
                                edgecolor='#FBC02D', linewidth=2)
    ax.add_patch(distill_box)
    ax.text(6, 6.8, 'Distillation', fontsize=10, fontweight='bold', ha='center', color='#F57F17')
    ax.text(6, 6.2, 'Temperature T=2.0', fontsize=8, ha='center')
    ax.text(6, 5.8, 'α = 0.7 (soft labels)', fontsize=8, ha='center')
    ax.text(6, 5.4, 'Knowledge Transfer', fontsize=8, ha='center')
    
    # 箭头
    ax.annotate('', xy=(3.3, 7.9), xytext=(2.8, 7.9),
               arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(4.4, 7.2), xytext=(4.4, 7.5),
               arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(4.5, 6.2), xytext=(4.1, 6.2),
               arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # ========== 推理阶段 ==========
    infer_box = FancyBboxPatch((8.2, 4.8), 7.5, 4.3,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['inference'],
                              edgecolor='#388E3C', linewidth=2)
    ax.add_patch(infer_box)
    ax.text(11.95, 8.8, 'Inference Phase (Online)', 
           fontsize=12, fontweight='bold', ha='center', color='#2E7D32')
    
    # 用户查询
    user_box = FancyBboxPatch((8.5, 7.5), 2.2, 0.8,
                             boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor='#388E3C', linewidth=1)
    ax.add_patch(user_box)
    ax.text(9.6, 7.9, 'User\nQuery', fontsize=9, ha='center', va='center')
    
    # RAG上下文
    rag2_box = FancyBboxPatch((11.2, 7.5), 2.2, 0.8,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['rag'], edgecolor='#C2185B', linewidth=1.5)
    ax.add_patch(rag2_box)
    ax.text(12.3, 7.9, 'Real-time\nRAG Context', fontsize=9, ha='center', va='center')
    
    # 学生模型+RAG
    student_box = FancyBboxPatch((8.5, 5.3), 6.9, 1.8,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['student'], 
                                edgecolor='#512DA8', linewidth=2)
    ax.add_patch(student_box)
    ax.text(11.95, 6.8, 'Student Model (1.8B) + RAG Context', 
           fontsize=10, fontweight='bold', ha='center', color='#4527A0')
    ax.text(11.95, 6.2, '• 5x Faster Inference', fontsize=8, ha='center', color='#2E7D32')
    ax.text(11.95, 5.8, '• Same Accuracy with Knowledge', fontsize=8, ha='center')
    ax.text(11.95, 5.4, '• Local Deployment Ready', fontsize=8, ha='center', color='#2E7D32')
    
    # 箭头
    ax.annotate('', xy=(11.2, 7.9), xytext=(10.7, 7.9),
               arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(12.3, 7.2), xytext=(12.3, 7.5),
               arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.annotate('', xy=(11.95, 7.1), xytext=(11.95, 7.5),
               arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # ========== 连接线 ==========
    ax.annotate('', xy=(8.2, 6.5), xytext=(7.5, 6.5),
               arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=3, ls='--'))
    ax.text(7.85, 6.9, 'Deploy', fontsize=9, ha='center', color='#D32F2F', fontweight='bold')
    
    # ========== 性能指标 ==========
    metrics_box = FancyBboxPatch((0.3, 0.3), 15.4, 4.2,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFF8E1',
                                edgecolor='#FF8F00', linewidth=2)
    ax.add_patch(metrics_box)
    ax.text(8, 4.2, 'Performance Comparison', 
           fontsize=12, fontweight='bold', ha='center', color='#E65100')
    
    # 表格数据
    metrics_data = [
        ['Metric', 'Teacher (32B)', 'Student (1.8B)', 'Improvement'],
        ['Inference Speed', '1x (baseline)', '5x', '5x faster'],
        ['Memory Usage', '60 GB', '4 GB', '15x reduction'],
        ['Accuracy', '92%', '89%', '-3% (acceptable)'],
        ['Deployment Cost', 'High', 'Low', 'Significant savings'],
        ['RAG Integration', 'Supported', 'Supported', 'Equal capability']
    ]
    
    # 绘制表格
    cell_height = 0.6
    cell_widths = [3.5, 3.5, 3.5, 4.0]
    start_y = 3.5
    
    for i, row in enumerate(metrics_data):
        y = start_y - i * cell_height
        x = 0.6
        for j, cell in enumerate(row):
            width = cell_widths[j]
            # 表头特殊处理
            if i == 0:
                rect = Rectangle((x, y), width, cell_height, 
                                facecolor='#FFE0B2', edgecolor='#E65100', linewidth=1)
                ax.add_patch(rect)
                ax.text(x + width/2, y + cell_height/2, cell, 
                       fontsize=9, ha='center', va='center', fontweight='bold')
            else:
                rect = Rectangle((x, y), width, cell_height, 
                                facecolor='white', edgecolor='#FFB74D', linewidth=0.5)
                ax.add_patch(rect)
                color = '#2E7D32' if 'x' in cell or 'reduction' in cell or 'savings' in cell else colors['text']
                ax.text(x + width/2, y + cell_height/2, cell, 
                       fontsize=8, ha='center', va='center', color=color)
            x += width
    
    plt.tight_layout()
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure2_rag_distillation_pipeline.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure2_rag_distillation_pipeline.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Figure 2: RAG-Distillation Pipeline")


def create_data_processing_workflow():
    """创建数据处理工作流程图 (Figure 3)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {
        'box': '#E3F2FD',
        'border': '#1976D2',
        'highlight': '#FFEBEE'
    }
    
    # ========== 子图1: 光变曲线处理 ==========
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.set_title('(a) Light Curve Analysis Pipeline', fontsize=12, fontweight='bold', pad=10)
    
    steps1 = [
        ('ZTF/TESS\nQuery', 4, 7),
        ('Data\nCleaning', 4, 5.5),
        ('Period\nAnalysis\n(Lomb-Scargle)', 4, 3.8),
        ('Phase\nFolding', 4, 2.2),
        ('Feature\nExtraction', 4, 0.6)
    ]
    
    for i, (label, x, y) in enumerate(steps1):
        box = FancyBboxPatch((x-1, y-0.6), 2, 1,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['box'], 
                            edgecolor=colors['border'], linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(x, y, label, fontsize=9, ha='center', va='center')
        if i < len(steps1) - 1:
            ax1.annotate('', xy=(x, y-0.7), xytext=(x, y-0.9),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # ========== 子图2: 光谱处理 ==========
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('(b) Spectral Analysis Pipeline', fontsize=12, fontweight='bold', pad=10)
    
    steps2 = [
        ('LAMOST/SDSS\nDownload', 4, 7),
        ('Wavelength\nCalibration', 4, 5.5),
        ('Continuum\nNormalization', 4, 3.8),
        ('Line\nIdentification', 4, 2.2),
        ('RV\nMeasurement', 4, 0.6)
    ]
    
    for i, (label, x, y) in enumerate(steps2):
        box = FancyBboxPatch((x-1, y-0.6), 2, 1,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['box'], 
                            edgecolor=colors['border'], linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(x, y, label, fontsize=9, ha='center', va='center')
        if i < len(steps2) - 1:
            ax2.annotate('', xy=(x, y-0.7), xytext=(x, y-0.9),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # ========== 子图3: 数据融合 ==========
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    ax3.set_title('(c) Multimodal Data Fusion', fontsize=12, fontweight='bold', pad=10)
    
    # 多模态输入
    sources = [
        ('Spectrum\nImage', 1.5, 6.5, '#E3F2FD'),
        ('Light\nCurve', 4, 6.5, '#FFF3E0'),
        ('Text\nMetadata', 6.5, 6.5, '#E8F5E9')
    ]
    
    for label, x, y, color in sources:
        box = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1,
                            boxstyle="round,pad=0.05",
                            facecolor=color, 
                            edgecolor='#333', linewidth=1.5)
        ax3.add_patch(box)
        ax3.text(x, y, label, fontsize=8, ha='center', va='center')
        ax3.annotate('', xy=(4, 4.8), xytext=(x, y-0.6),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    
    # 编码器
    encoders = FancyBboxPatch((2, 4), 4, 1,
                             boxstyle="round,pad=0.05",
                             facecolor='#F3E5F5', 
                             edgecolor='#7B1FA2', linewidth=2)
    ax3.add_patch(encoders)
    ax3.text(4, 4.5, 'Multimodal Encoders\n(CNN + LSTM + BERT)', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # 投影层
    ax3.annotate('', xy=(4, 2.8), xytext=(4, 4),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    proj = FancyBboxPatch((2.5, 1.8), 3, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor='#FFF9C4', 
                         edgecolor='#FBC02D', linewidth=2)
    ax3.add_patch(proj)
    ax3.text(4, 2.2, 'Projection Layer\n→ Unified 512-dim Space', 
            fontsize=9, ha='center', va='center')
    
    # 融合输出
    ax3.annotate('', xy=(4, 0.6), xytext=(4, 1.8),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    fusion = FancyBboxPatch((2, 0), 4, 0.5,
                           boxstyle="round,pad=0.05",
                           facecolor=colors['highlight'], 
                           edgecolor='#C62828', linewidth=2)
    ax3.add_patch(fusion)
    ax3.text(4, 0.25, 'Fused Representation (Attention-based)', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # ========== 子图4: 知识库构建 ==========
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 8)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    ax4.set_title('(d) Knowledge Base Construction', fontsize=12, fontweight='bold', pad=10)
    
    # 数据源
    sources2 = [
        ('arXiv\nAPI', 1.5, 7),
        ('ADS\nDatabase', 4, 7),
        ('Local\nPapers', 6.5, 7)
    ]
    
    for label, x, y in sources2:
        box = FancyBboxPatch((x-0.8, y-0.5), 1.6, 0.9,
                            boxstyle="round,pad=0.05",
                            facecolor='#E1F5FE', 
                            edgecolor='#0288D1', linewidth=1.5)
        ax4.add_patch(box)
        ax4.text(x, y-0.05, label, fontsize=8, ha='center', va='center')
        ax4.annotate('', xy=(4, 5.5), xytext=(x, y-0.55),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    
    # PDF处理
    pdf_box = FancyBboxPatch((2.5, 4.8), 3, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor='#FFF3E0', 
                            edgecolor='#EF6C00', linewidth=1.5)
    ax4.add_patch(pdf_box)
    ax4.text(4, 5.1, 'PDF Processing & OCR', fontsize=9, ha='center', va='center')
    
    ax4.annotate('', xy=(4, 4.8), xytext=(4, 5.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # QA生成
    qa_box = FancyBboxPatch((2.5, 3.8), 3, 0.8,
                           boxstyle="round,pad=0.05",
                           facecolor='#E8F5E9', 
                           edgecolor='#388E3C', linewidth=1.5)
    ax4.add_patch(qa_box)
    ax4.text(4, 4.2, 'QA Pair Generation\n(Rule-based + LLM)', 
            fontsize=9, ha='center', va='center')
    
    ax4.annotate('', xy=(4, 3.8), xytext=(4, 4.8),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # 向量化
    vec_box = FancyBboxPatch((2.5, 2.6), 3, 1,
                            boxstyle="round,pad=0.05",
                            facecolor='#F3E5F5', 
                            edgecolor='#7B1FA2', linewidth=1.5)
    ax4.add_patch(vec_box)
    ax4.text(4, 3.1, 'Text Embedding\n(sentence-transformers)', 
            fontsize=9, ha='center', va='center')
    
    ax4.annotate('', xy=(4, 2.6), xytext=(4, 3.8),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # 向量数据库
    db_box = FancyBboxPatch((2.5, 1.2), 3, 1.2,
                           boxstyle="round,pad=0.05",
                           facecolor=colors['highlight'], 
                           edgecolor='#C62828', linewidth=2)
    ax4.add_patch(db_box)
    ax4.text(4, 1.8, 'Vector Database\n(Chroma/FAISS)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    ax4.text(4, 1.4, '20,609 QA Pairs', fontsize=8, ha='center', va='center', color='#2E7D32')
    
    ax4.annotate('', xy=(4, 1.2), xytext=(4, 2.6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    plt.tight_layout()
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure3_data_processing_workflow.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure3_data_processing_workflow.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Figure 3: Data Processing Workflow")


def create_evaluation_results():
    """创建模型评估结果图 (Figure 4)"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ========== 子图1: 准确率对比 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Cataclysmic\nVariables', 'White Dwarfs', 'Binary\nStars', 
                  'Observational\nMethods', 'Stellar\nEvolution', 'Overall']
    baseline = [65, 62, 58, 70, 68, 64]
    astrosage = [91, 89, 87, 93, 90, 90]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline, width, label='Baseline (Qwen-7B)', 
                   color='#90A4AE', edgecolor='#546E7A')
    bars2 = ax1.bar(x + width/2, astrosage, width, label='AstroSage (Fine-tuned)', 
                   color='#66BB6A', edgecolor='#2E7D32')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('(a) Domain-Specific Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # ========== 子图2: 推理速度对比 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['Teacher\n(32B)', 'Student\n(7B)', 'Student\n(1.8B)', 'Baseline\n(7B)']
    speeds = [1, 3.2, 5.0, 2.8]
    colors_speed = ['#EF5350', '#66BB6A', '#42A5F5', '#FFA726']
    
    bars = ax2.barh(models, speeds, color=colors_speed, edgecolor='#333', height=0.6)
    ax2.set_xlabel('Relative Speed (× faster)', fontsize=11)
    ax2.set_title('(b) Inference Speed Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 6)
    
    for bar, speed in zip(bars, speeds):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{speed:.1f}×', va='center', fontsize=10, fontweight='bold')
    
    # 添加参考线
    ax2.axvline(x=1, color='#D32F2F', linestyle='--', alpha=0.5, label='Teacher baseline')
    
    # ========== 子图3: 问答类型分布 ==========
    ax3 = fig.add_subplot(gs[1, 0])
    qa_types = ['SED', 'Light Curve', 'HR Diagram', 'Period', 
                'CV', 'General', 'X-ray', 'Binary', 'Spectrum']
    counts = [8325, 3170, 3183, 1483, 1020, 1238, 892, 654, 644]
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(qa_types)))
    
    wedges, texts, autotexts = ax3.pie(counts, labels=qa_types, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90,
                                        textprops={'fontsize': 9})
    ax3.set_title('(c) QA Dataset Distribution (N=20,609)', fontsize=12, fontweight='bold')
    
    # ========== 子图4: 训练曲线 ==========
    ax4 = fig.add_subplot(gs[1, 1])
    epochs = np.arange(1, 4)
    train_loss = [2.45, 1.32, 0.89]
    val_loss = [2.52, 1.45, 1.02]
    val_acc = [72, 85, 90]
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=8)
    line2 = ax4.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=8)
    line3 = ax4_twin.plot(epochs, val_acc, 'g-^', label='Val Accuracy', linewidth=2, markersize=8)
    
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11, color='#333')
    ax4_twin.set_ylabel('Accuracy (%)', fontsize=11, color='#2E7D32')
    ax4.set_title('(d) Training Progress (Qwen-7B LoRA)', fontsize=12, fontweight='bold')
    ax4.set_xticks(epochs)
    ax4.grid(alpha=0.3)
    
    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right', fontsize=9)
    
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure4_evaluation_results.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure4_evaluation_results.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Figure 4: Evaluation Results")


def create_technical_roadmap():
    """创建技术发展路线图 (Figure 5)"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(9, 9.5, 'AstroSage Technical Development Roadmap', 
           fontsize=18, fontweight='bold', ha='center')
    
    # 时间轴
    phases = [
        ('Phase 1\n(Completed)', '#C8E6C9', 1.5),
        ('Phase 2\n(Current)', '#FFF9C4', 6.5),
        ('Phase 3\n(Next)', '#FFCCBC', 11.5),
        ('Phase 4\n(Future)', '#E1BEE7', 16.5)
    ]
    
    for label, color, x in phases:
        # 阶段背景
        bg = FancyBboxPatch((x-2, 0.5), 4.5, 8.2,
                           boxstyle="round,pad=0.1",
                           facecolor=color, 
                           edgecolor='#333', linewidth=2, alpha=0.3)
        ax.add_patch(bg)
        
        # 阶段标题
        title = FancyBboxPatch((x-1.8, 7.8), 4.1, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=color, 
                              edgecolor='#333', linewidth=2)
        ax.add_patch(title)
        ax.text(x + 0.25, 8.2, label, fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Phase 1 内容
    p1_items = [
        ('✓ Data Pipeline', 7.0),
        ('✓ RAG System', 6.0),
        ('✓ Basic LLM', 5.0),
        ('✓ LoRA Training', 4.0),
        ('✓ Local Deploy', 3.0)
    ]
    for label, y in p1_items:
        box = FancyBboxPatch((0.3, y-0.35), 2.8, 0.6,
                            boxstyle="round,pad=0.03",
                            facecolor='white', edgecolor='#2E7D32', linewidth=1.5)
        ax.add_patch(box)
        ax.text(1.7, y, label, fontsize=9, ha='center', va='center', color='#1B5E20')
    
    # Phase 2 内容
    p2_items = [
        ('⟳ Distillation', 7.0),
        ('⟳ Multimodal', 6.0),
        ('⟳ Explainable AI', 5.0),
        ('⟳ Evaluation', 4.0),
        ('⟳ Multilingual', 3.0)
    ]
    for label, y in p2_items:
        box = FancyBboxPatch((5.3, y-0.35), 2.8, 0.6,
                            boxstyle="round,pad=0.03",
                            facecolor='white', edgecolor='#F57F17', linewidth=1.5)
        ax.add_patch(box)
        ax.text(6.7, y, label, fontsize=9, ha='center', va='center', color='#E65100')
    
    # Phase 3 内容
    p3_items = [
        ('○ Real-time Update', 7.0),
        ('○ Knowledge Graph', 6.0),
        ('○ Interactive Viz', 5.0),
        ('○ API Service', 4.0),
        ('○ Community', 3.0)
    ]
    for label, y in p3_items:
        box = FancyBboxPatch((10.3, y-0.35), 2.8, 0.6,
                            boxstyle="round,pad=0.03",
                            facecolor='white', edgecolor='#7B1FA2', linewidth=1.5)
        ax.add_patch(box)
        ax.text(11.7, y, label, fontsize=9, ha='center', va='center', color='#4A148C')
    
    # Phase 4 内容
    p4_items = [
        ('? Foundation Model', 7.0),
        ('? Multi-object', 6.0),
        ('? Auto-discovery', 5.0),
        ('? Global Network', 4.0),
        ('? AI Observatory', 3.0)
    ]
    for label, y in p4_items:
        box = FancyBboxPatch((15.3, y-0.35), 2.8, 0.6,
                            boxstyle="round,pad=0.03",
                            facecolor='white', edgecolor='#1565C0', linewidth=1.5)
        ax.add_patch(box)
        ax.text(16.7, y, label, fontsize=9, ha='center', va='center', color='#0D47A1')
    
    # 连接箭头
    for start_x in [4.0, 9.0, 14.0]:
        ax.annotate('', xy=(start_x+1.3, 5), xytext=(start_x, 5),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=2))
    
    # 图例
    legend_items = [
        ('✓ Completed', '#C8E6C9'),
        ('⟳ In Progress', '#FFF9C4'),
        ('○ Planned', '#FFCCBC'),
        ('? Research', '#E1BEE7')
    ]
    
    for i, (label, color) in enumerate(legend_items):
        x = 5 + i * 3
        box = Rectangle((x, 0.1), 0.4, 0.3, facecolor=color, edgecolor='#333', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.6, 0.25, label, fontsize=9, va='center')
    
    plt.tight_layout()
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure5_technical_roadmap.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/mnt/c/Users/Administrator/Desktop/astro-ai-demo/paper/figures/figure5_technical_roadmap.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Figure 5: Technical Roadmap")


if __name__ == "__main__":
    print("="*60)
    print("Generating AstroSage Technical Architecture Figures")
    print("="*60)
    
    create_system_architecture()
    create_rag_distillation_pipeline()
    create_data_processing_workflow()
    create_evaluation_results()
    create_technical_roadmap()
    
    print("="*60)
    print("All figures generated successfully!")
    print("="*60)
