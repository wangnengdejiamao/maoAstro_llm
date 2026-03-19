#!/usr/bin/env python3
"""
天文学QA模型训练器
使用简单的TF-IDF + 余弦相似度方法
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

class AstroQAModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b'
        )
        self.questions = []
        self.answers = []
        self.topics = []
        self.question_vectors = None
        
    def preprocess_text(self, text):
        """文本预处理"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def train(self, train_data):
        """训练模型"""
        print(f"训练模型，使用 {len(train_data)} 个样本...")
        
        self.questions = [self.preprocess_text(item['question']) for item in train_data]
        self.answers = [item['answer'] for item in train_data]
        self.topics = [item.get('topic', '') for item in train_data]
        
        # 创建TF-IDF向量
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        print(f"词汇表大小: {len(self.vectorizer.vocabulary_)}")
        
        return self
    
    def predict(self, query, top_k=3):
        """预测答案"""
        query_processed = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([query_processed])
        
        # 计算相似度
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        
        # 获取top-k最相似的问题
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'topic': self.topics[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def save(self, model_path):
        """保存模型"""
        model_data = {
            'vectorizer': self.vectorizer,
            'questions': self.questions,
            'answers': self.answers,
            'topics': self.topics,
            'question_vectors': self.question_vectors
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到 {model_path}")
    
    def load(self, model_path):
        """加载模型"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.vectorizer = model_data['vectorizer']
        self.questions = model_data['questions']
        self.answers = model_data['answers']
        self.topics = model_data['topics']
        self.question_vectors = model_data['question_vectors']
        print(f"模型已从 {model_path} 加载")
        return self

def train_and_save():
    """训练并保存模型"""
    # 加载训练数据
    with open('output/astro_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 训练模型
    model = AstroQAModel()
    model.train(train_data)
    
    # 保存模型
    Path('models').mkdir(exist_ok=True)
    model.save('models/astro_qa_model.pkl')
    
    return model

if __name__ == "__main__":
    model = train_and_save()
    
    # 测试几个例子
    print("\n=== 模型测试 ===")
    test_queries = [
        "什么是灾变变星？",
        "白矮星的极限质量是多少？",
        "ZTF是什么望远镜？",
        "什么是吸积盘？"
    ]
    
    for query in test_queries:
        print(f"\n问题: {query}")
        results = model.predict(query, top_k=1)
        if results:
            print(f"答案: {results[0]['answer'][:200]}...")
            print(f"相似度: {results[0]['similarity']:.3f}")
