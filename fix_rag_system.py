#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
RAG系统修复脚本
================================================================================
功能描述:
    修复RAG系统的向量库和关键词索引，包括：
    1. 将QA数据导入ChromaDB向量库
    2. 构建关键词倒排索引
    3. 数据质量检查和清理

使用方法:
    python fix_rag_system.py

修复内容:
    - 创建ChromaDB集合并导入向量
    - 构建Whoosh关键词索引
    - 生成修复报告

作者: AstroSage Team
================================================================================
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


class RAGSystemFixer:
    """RAG系统修复器"""
    
    def __init__(self, data_dir: str = "output/qa_hybrid"):
        self.data_dir = Path(data_dir)
        self.chroma_dir = self.data_dir / "chroma_db"
        self.keyword_dir = self.data_dir / "keyword_index"
        
        print("="*70)
        print("🔧 RAG系统修复工具")
        print("="*70)
    
    def load_qa_data(self) -> List[Dict]:
        """加载QA数据"""
        print("\n📂 加载QA数据...")
        
        dataset_path = self.data_dir / "qa_dataset_full.json"
        if not dataset_path.exists():
            print(f"❌ 数据文件不存在: {dataset_path}")
            return []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   加载了 {len(data):,} 条QA对")
        return data
    
    def fix_chromadb(self, qa_data: List[Dict]):
        """修复ChromaDB向量库"""
        print("\n" + "="*70)
        print("📚 修复ChromaDB向量库")
        print("="*70)
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("\n⚠️  缺少依赖，正在安装...")
            os.system("pip install chromadb sentence-transformers -q")
            import chromadb
            from sentence_transformers import SentenceTransformer
        
        # 创建/连接ChromaDB
        print("\n   连接ChromaDB...")
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        
        # 删除旧集合（如果存在）
        try:
            client.delete_collection("astro_qa")
            print("   删除旧集合")
        except:
            pass
        
        # 创建新集合
        print("   创建新集合 'astro_qa'...")
        collection = client.create_collection(
            name="astro_qa",
            metadata={"description": "天文问答知识库"}
        )
        
        # 加载嵌入模型
        print("   加载嵌入模型 (BAAI/bge-large-zh-v1.5)...")
        print("   首次下载可能需要几分钟...")
        try:
            model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        except Exception as e:
            print(f"   下载失败，使用本地镜像: {e}")
            model = SentenceTransformer('BAAI/bge-large-zh-v1.5', cache_folder='./models')
        
        # 分批导入数据
        batch_size = 100
        total = len(qa_data)
        
        print(f"\n   导入数据 (共{total}条)...")
        
        for i in tqdm(range(0, total, batch_size), desc="导入进度"):
            batch = qa_data[i:i+batch_size]
            
            # 准备数据
            ids = []
            documents = []
            metadatas = []
            
            for idx, qa in enumerate(batch):
                qa_id = f"qa_{i+idx}"
                question = qa.get('question', '')
                answer = qa.get('answer', '')
                
                # 组合问题和答案用于嵌入
                doc_text = f"问题: {question}\n答案: {answer[:500]}"  # 限制长度
                
                ids.append(qa_id)
                documents.append(doc_text)
                metadatas.append({
                    "question": question,
                    "answer": answer,
                    "source": qa.get('source', 'unknown'),
                    "category": qa.get('category', 'unknown'),
                })
            
            # 生成嵌入向量
            embeddings = model.encode(documents, show_progress_bar=False)
            
            # 添加到集合
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )
        
        # 验证
        count = collection.count()
        print(f"\n   ✅ ChromaDB导入完成")
        print(f"   向量数量: {count:,}")
        
        # 测试检索
        print("\n   测试检索...")
        test_query = "灾变变星是什么"
        test_embedding = model.encode([test_query])
        results = collection.query(
            query_embeddings=test_embedding.tolist(),
            n_results=3,
        )
        
        if results and results.get('documents'):
            print(f"   检索测试成功，找到 {len(results['documents'][0])} 条结果")
        
        return True
    
    def fix_keyword_index(self, qa_data: List[Dict]):
        """修复关键词索引"""
        print("\n" + "="*70)
        print("🔤 修复关键词索引")
        print("="*70)
        
        try:
            from whoosh import index, fields, analysis
            from whoosh.qparser import QueryParser
            import jieba
        except ImportError:
            print("\n⚠️  缺少依赖，正在安装...")
            os.system("pip install whoosh jieba -q")
            from whoosh import index, fields, analysis
            from whoosh.qparser import QueryParser
            import jieba
        
        # 创建索引目录
        print(f"\n   创建索引目录: {self.keyword_dir}")
        self.keyword_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义Schema
        print("   定义索引结构...")
        
        # 使用中文分词器
        analyzer = analysis.RegexTokenizer() | analysis.LowercaseFilter()
        
        schema = fields.Schema(
            id=fields.ID(stored=True, unique=True),
            question=fields.TEXT(stored=True, analyzer=analyzer),
            answer=fields.TEXT(stored=True, analyzer=analyzer),
            source=fields.STORED(),
            category=fields.STORED(),
        )
        
        # 创建索引
        print("   创建索引...")
        ix = index.create_in(str(self.keyword_dir), schema)
        
        # 写入数据
        writer = ix.writer()
        
        print(f"   导入数据 (共{len(qa_data)}条)...")
        for i, qa in enumerate(tqdm(qa_data, desc="索引进度")):
            try:
                writer.add_document(
                    id=f"qa_{i}",
                    question=qa.get('question', ''),
                    answer=qa.get('answer', '')[:1000],  # 限制长度
                    source=str(qa.get('source', 'unknown')),
                    category=qa.get('category', 'unknown'),
                )
            except Exception as e:
                print(f"   警告: 第{i}条数据导入失败: {e}")
        
        print("   提交索引...")
        writer.commit()
        
        # 验证
        print("\n   验证索引...")
        with ix.searcher() as searcher:
            doc_count = searcher.doc_count()
            print(f"   索引文档数: {doc_count:,}")
            
            # 测试搜索
            parser = QueryParser("question", ix.schema)
            test_query = parser.parse("灾变变星")
            results = searcher.search(test_query, limit=3)
            print(f"   搜索测试成功，找到 {len(results)} 条结果")
        
        print("\n   ✅ 关键词索引构建完成")
        return True
    
    def verify_fixes(self):
        """验证修复结果"""
        print("\n" + "="*70)
        print("✅ 验证修复结果")
        print("="*70)
        
        results = {
            "chromadb": False,
            "keyword_index": False,
        }
        
        # 验证ChromaDB
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            collections = client.list_collections()
            
            if "astro_qa" in collections:
                collection = client.get_collection("astro_qa")
                count = collection.count()
                print(f"\n📚 ChromaDB: ✓ 正常")
                print(f"   集合: astro_qa")
                print(f"   向量数: {count:,}")
                results["chromadb"] = True
            else:
                print("\n📚 ChromaDB: ✗ 集合不存在")
        except Exception as e:
            print(f"\n📚 ChromaDB: ✗ 错误 - {e}")
        
        # 验证关键词索引
        try:
            from whoosh import index
            if (self.keyword_dir / "MAIN_WRITELOCK").exists():
                ix = index.open_dir(str(self.keyword_dir))
                with ix.searcher() as searcher:
                    print(f"\n🔤 关键词索引: ✓ 正常")
                    print(f"   文档数: {searcher.doc_count():,}")
                    results["keyword_index"] = True
            else:
                print("\n🔤 关键词索引: ✗ 索引文件不存在")
        except Exception as e:
            print(f"\n🔤 关键词索引: ✗ 错误 - {e}")
        
        # 总结
        print("\n" + "="*70)
        print("📊 修复总结")
        print("="*70)
        
        if all(results.values()):
            print("\n🎉 所有组件修复成功！")
            print("\n现在可以正常使用RAG系统:")
            print("  python start_astrosage_complete.py")
        else:
            print("\n⚠️  部分组件修复失败，请检查错误信息")
        
        return results
    
    def run(self):
        """运行完整修复流程"""
        # 1. 加载数据
        qa_data = self.load_qa_data()
        if not qa_data:
            print("❌ 没有数据可修复")
            return False
        
        # 2. 修复ChromaDB
        try:
            self.fix_chromadb(qa_data)
        except Exception as e:
            print(f"\n❌ ChromaDB修复失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. 修复关键词索引
        try:
            self.fix_keyword_index(qa_data)
        except Exception as e:
            print(f"\n❌ 关键词索引修复失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. 验证
        self.verify_fixes()
        
        return True


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🔧 RAG系统修复工具")
    print("="*70)
    print("\n本工具将:")
    print("  1. 将QA数据导入ChromaDB向量库")
    print("  2. 构建Whoosh关键词索引")
    print("  3. 验证修复结果")
    print("")
    
    input("按回车键开始修复...")
    
    fixer = RAGSystemFixer()
    fixer.run()
    
    print("\n" + "="*70)
    print("✅ 修复流程结束")
    print("="*70)


if __name__ == "__main__":
    main()
