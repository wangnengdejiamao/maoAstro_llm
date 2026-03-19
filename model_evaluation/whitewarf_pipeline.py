#!/usr/bin/env python3
"""
WhiteWarf 完整流程控制器

整合所有模块的完整流程：
1. 扫描 F:\storage 的 PDF
2. 去重和提取
3. Kimi 数据提取和 RAG 构建
4. 模型微调
5. 评估和可视化
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional


def setup_environment():
    """设置环境"""
    print("=" * 70)
    print("🌟 WhiteWarf: Qwen-8B 白矮星专用模型 - 完整流程")
    print("=" * 70)
    
    # 检查 Python 版本
    import sys
    if sys.version_info < (3, 8):
        print("❌ 需要 Python 3.8+")
        sys.exit(1)
    
    # 创建必要的目录
    dirs = [
        "./pdf_library",
        "./rag_knowledge_base",
        "./models",
        "./data/white_dwarf_papers",
        "./interview_charts",
        "./eval_results"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print("✓ 环境检查完成")


def step1_scan_pdfs(pdf_dir: str = r"F:\storage"):
    """步骤 1: 扫描 PDF"""
    print("\n" + "=" * 70)
    print("步骤 1/6: 扫描 PDF 文件")
    print("=" * 70)
    
    from pdf_processor import PDFScanner, PDFDeduplicator
    
    scanner = PDFScanner(pdf_dir)
    pdf_files = scanner.scan()
    
    if not pdf_files:
        print("⚠ 未找到 PDF 文件，请检查路径")
        return None
    
    # 去重
    print("\n去重处理...")
    deduplicator = PDFDeduplicator()
    unique_files, duplicate_files = deduplicator.deduplicate(pdf_files)
    
    # 保存索引
    deduplicator.save_index(unique_files, "./pdf_library/unique_pdfs.json")
    deduplicator.save_index(duplicate_files, "./pdf_library/duplicate_pdfs.json")
    
    print(f"\n✓ 找到 {len(unique_files)} 个唯一 PDF 文件")
    print(f"✓ 发现 {len(duplicate_files)} 个重复文件")
    
    return unique_files


def step2_extract_data(unique_files, use_kimi: bool = False):
    """步骤 2: 提取数据"""
    print("\n" + "=" * 70)
    print("步骤 2/6: 提取 PDF 数据")
    print("=" * 70)
    
    from pdf_processor import PDFTextExtractor
    from kimi_pdf_extractor import KimiPDFExtractor, AstroSourceDatabase
    
    text_extractor = PDFTextExtractor()
    
    if use_kimi:
        print("使用 Kimi API 提取结构化数据...")
        kimi_extractor = KimiPDFExtractor()
    else:
        print("使用模板方法提取数据...")
        kimi_extractor = None
    
    source_db = AstroSourceDatabase()
    all_sources = []
    
    # 限制处理数量（测试时用）
    max_process = min(len(unique_files), 20)
    
    for i, pdf_info in enumerate(unique_files[:max_process], 1):
        print(f"\n[{i}/{max_process}] {pdf_info.filename[:50]}")
        
        try:
            # 提取文本
            text = text_extractor.extract_text(pdf_info.filepath, max_pages=25)
            
            if len(text) < 200:
                print("  ⚠ 文本过短，跳过")
                continue
            
            # 保存提取的文本
            text_path = Path("./pdf_library/extracted_text") / f"{pdf_info.filename}.txt"
            text_path.parent.mkdir(exist_ok=True)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # 使用 Kimi 提取结构化数据
            if use_kimi and kimi_extractor:
                try:
                    summary = kimi_extractor.extract_paper_summary(text)
                    sources = kimi_extractor.extract_astro_sources(text)
                    
                    for source in sources:
                        source_db.add_source(source)
                        all_sources.append(source)
                    
                    if sources:
                        print(f"  ✓ 提取 {len(sources)} 个天体源")
                except Exception as e:
                    print(f"  ⚠ Kimi 提取失败: {e}")
            
            print(f"  ✓ 提取 {len(text)} 字符")
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    # 保存数据库
    source_db.save()
    
    print(f"\n✓ 共提取 {len(all_sources)} 个天体源")
    
    return all_sources


def step3_build_rag(unique_files, use_kimi: bool = False):
    """步骤 3: 构建 RAG 知识库"""
    print("\n" + "=" * 70)
    print("步骤 3/6: 构建 RAG 知识库")
    print("=" * 70)
    
    from pdf_processor import PDFTextExtractor
    from kimi_pdf_extractor import KimiPDFExtractor
    from rag_knowledge_base import RAGKnowledgeBase, RAGDocument
    
    text_extractor = PDFTextExtractor()
    kimi_extractor = KimiPDFExtractor() if use_kimi else None
    kb = RAGKnowledgeBase()
    
    max_process = min(len(unique_files), 15)
    
    for i, pdf_info in enumerate(unique_files[:max_process], 1):
        print(f"\n[{i}/{max_process}] 处理: {pdf_info.filename[:40]}")
        
        try:
            # 提取文本
            text = text_extractor.extract_text(pdf_info.filepath, max_pages=20)
            
            if len(text) < 200:
                continue
            
            # 创建知识块
            if use_kimi and kimi_extractor:
                chunks = kimi_extractor.create_knowledge_chunks(text, pdf_info.filename)
            else:
                # 简单的分块
                chunks = []
                chunk_size = 1500
                overlap = 300
                start = 0
                chunk_idx = 0
                
                while start < len(text):
                    end = start + chunk_size
                    chunk_text = text[start:end]
                    
                    chunk = RAGDocument(
                        doc_id=f"{pdf_info.filename}_chunk_{chunk_idx}",
                        content=chunk_text,
                        metadata={"source_pdf": pdf_info.filename, "chunk_type": "text"},
                        embedding=None
                    )
                    chunks.append(chunk)
                    
                    start = end - overlap
                    chunk_idx += 1
            
            # 添加到知识库
            kb.add_documents(chunks, compute_embeddings=False)  # 延迟计算嵌入
            
            print(f"  ✓ 创建 {len(chunks)} 个知识块")
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    # 计算所有嵌入
    print("\n计算嵌入向量...")
    if kb.documents:
        kb._compute_embeddings(kb.documents)
    
    # 保存知识库
    kb.save()
    
    # 统计
    stats = kb.get_statistics()
    print(f"\n✓ RAG 知识库构建完成")
    print(f"  总文档: {stats['total_documents']}")
    print(f"  嵌入维度: {stats['embedding_dim']}")
    
    return kb


def step4_train_model(use_kimi: bool = False):
    """步骤 4: 训练模型"""
    print("\n" + "=" * 70)
    print("步骤 4/6: 训练模型")
    print("=" * 70)
    
    print("选择训练方式:")
    print("  1. 使用模板数据快速训练（推荐）")
    print("  2. 使用 Kimi 生成数据训练")
    
    if use_kimi:
        script = "train_whitewarf.sh"
    else:
        script = "train_whitewarf_no_kimi.sh"
    
    print(f"\n运行训练脚本: {script}")
    print("注意: 训练需要 GPU 和较长时间，请确保资源充足")
    
    # 这里只是提示，实际训练需要手动运行
    print(f"\n请手动运行: bash {script}")
    
    return True


def step5_evaluate():
    """步骤 5: 评估模型"""
    print("\n" + "=" * 70)
    print("步骤 5/6: 模型评估")
    print("=" * 70)
    
    print("评估选项:")
    print("  1. 白矮星专用评估")
    print("  2. 对比 Llama-3.1-8B")
    
    # 提示用户运行评估脚本
    print("\n请运行以下命令进行评估:")
    print("  python wd_evaluator.py --model your-model --interface hf")
    
    return True


def step6_visualize():
    """步骤 6: 生成可视化"""
    print("\n" + "=" * 70)
    print("步骤 6/6: 生成面试展示材料")
    print("=" * 70)
    
    print("生成可视化图表和面试报告...")
    
    from evaluate_and_visualize import main as viz_main
    viz_main()
    
    return True


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description="WhiteWarf 完整流程")
    parser.add_argument("--pdf-dir", type=str, default=r"F:\storage", 
                       help="PDF 目录路径")
    parser.add_argument("--use-kimi", action="store_true",
                       help="使用 Kimi API")
    parser.add_argument("--step", type=int, default=0,
                       help="从指定步骤开始（0=全部）")
    parser.add_argument("--skip-training", action="store_true",
                       help="跳过训练步骤")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 检查 PDF 目录
    if not Path(args.pdf_dir).exists():
        print(f"\n⚠ PDF 目录不存在: {args.pdf_dir}")
        print("请检查路径或提供正确的 --pdf-dir 参数")
        return
    
    # 执行流程
    unique_files = None
    
    # 步骤 1: 扫描 PDF
    if args.step <= 1:
        unique_files = step1_scan_pdfs(args.pdf_dir)
        if not unique_files:
            print("\n❌ 未找到 PDF 文件，流程终止")
            return
    
    # 步骤 2: 提取数据
    if args.step <= 2:
        if unique_files is None:
            # 加载已有索引
            from pdf_processor import PDFDeduplicator
            dedup = PDFDeduplicator()
            if Path("./pdf_library/unique_pdfs.json").exists():
                unique_files = dedup.load_index("./pdf_library/unique_pdfs.json")
                print(f"加载 {len(unique_files)} 个已索引的 PDF")
        
        if unique_files:
            step2_extract_data(unique_files, args.use_kimi)
    
    # 步骤 3: 构建 RAG
    if args.step <= 3:
        if unique_files is None and Path("./pdf_library/unique_pdfs.json").exists():
            from pdf_processor import PDFDeduplicator
            dedup = PDFDeduplicator()
            unique_files = dedup.load_index("./pdf_library/unique_pdfs.json")
        
        if unique_files:
            step3_build_rag(unique_files, args.use_kimi)
    
    # 步骤 4: 训练模型
    if args.step <= 4 and not args.skip_training:
        step4_train_model(args.use_kimi)
    
    # 步骤 5: 评估
    if args.step <= 5:
        step5_evaluate()
    
    # 步骤 6: 可视化
    if args.step <= 6:
        step6_visualize()
    
    print("\n" + "=" * 70)
    print("✅ 流程执行完成!")
    print("=" * 70)
    print("\n生成的重要文件:")
    print("  📁 ./pdf_library/ - PDF 索引和提取的文本")
    print("  📁 ./rag_knowledge_base/ - RAG 知识库")
    print("  📁 ./models/ - 训练好的模型")
    print("  📁 ./interview_charts/ - 面试展示图表")
    print("\n祝你面试成功! 🎉")


if __name__ == "__main__":
    main()
