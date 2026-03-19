#!/usr/bin/env python3
"""
天文学QA模型演示
"""

from train_model import AstroQAModel

def main():
    print("="*60)
    print("  天体物理智能问答系统")
    print("="*60)
    print("\n正在加载模型...")
    
    model = AstroQAModel()
    model.load('models/astro_qa_model.pkl')
    
    print("模型加载完成！可以开始提问了。\n")
    print("支持的领域:")
    print("  - 灾变变星 (Cataclysmic Variables)")
    print("  - 白矮星 (White Dwarfs)")
    print("  - 双星演化 (Binary Evolution)")
    print("  - 吸积物理 (Accretion Physics)")
    print("  - 观测方法 (Observational Methods)")
    print("  - 恒星演化 (Stellar Evolution)")
    print("  - X射线天文学 (X-ray Astronomy)")
    print("\n输入 'quit' 退出，输入 'test' 运行测试\n")
    
    while True:
        query = input("> ").strip()
        
        if query.lower() == 'quit':
            print("再见！")
            break
        
        if query.lower() == 'test':
            run_tests(model)
            continue
        
        if not query:
            continue
        
        # 获取答案
        results = model.predict(query, top_k=3)
        
        if not results or results[0]['similarity'] < 0.1:
            print("抱歉，我暂时无法回答这个问题。请尝试用不同的方式提问。")
            print("例如：'什么是灾变变星？' 或 '什么是吸积盘？'\n")
            continue
        
        print("\n【回答】")
        print(results[0]['answer'])
        print(f"\n[置信度: {results[0]['similarity']:.2f} | 主题: {results[0]['topic']}]\n")
        
        # 显示其他可能答案
        if len(results) > 1 and results[1]['similarity'] > 0.1:
            print("【相关回答】")
            for i, r in enumerate(results[1:3], 2):
                if r['similarity'] > 0.1:
                    print(f"{i}. {r['answer'][:100]}... [相似度: {r['similarity']:.2f}]")
            print()

def run_tests(model):
    """运行测试题目"""
    test_questions = [
        "什么是灾变变星？",
        "白矮星是如何形成的？",
        "什么是吸积盘？",
        "ZTF是什么望远镜？",
        "什么是洛希瓣？",
        "Ia型超新星的前身星是什么？",
        "什么是赫罗图？",
        "什么是X射线双星？",
    ]
    
    print("\n" + "="*60)
    print("运行测试题目")
    print("="*60 + "\n")
    
    for q in test_questions:
        print(f"Q: {q}")
        results = model.predict(q, top_k=1)
        if results:
            print(f"A: {results[0]['answer'][:200]}...")
            print(f"   [置信度: {results[0]['similarity']:.2f} | 主题: {results[0]['topic']}]")
        print()

if __name__ == "__main__":
    main()
