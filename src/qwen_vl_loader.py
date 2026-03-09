#!/usr/bin/env python3
"""
Qwen-VL-8B 模型加载器
===================
针对 RTX 3080 Ti 12GB 显存优化

使用方法:
    python qwen_vl_loader.py --mode chat
    python qwen_vl_loader.py --image path/to/image.jpg
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 模型配置
# 优先使用本地模型，如果没有则使用HuggingFace/ModelScope
LOCAL_MODEL_PATH = "./models/qwen/Qwen-VL-Chat-Int4"  # ModelScope下载路径
HF_MODEL_NAME = "Qwen/Qwen-VL-Chat-Int4"  # HuggingFace模型名

# 自动检测使用哪个路径
if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_NAME = LOCAL_MODEL_PATH
    print(f"✓ 使用本地模型: {MODEL_NAME}")
else:
    MODEL_NAME = HF_MODEL_NAME
    print(f"⚠ 使用远程模型: {MODEL_NAME} (需要下载)")
    print(f"  建议先运行: python download_qwen_modelscope.py")

class QwenVLAgent:
    """Qwen-VL-8B 智能体"""
    
    def __init__(self, model_name=None, load_in_4bit=True):
        """
        初始化模型
        
        Args:
            model_name: HuggingFace模型名称
            load_in_4bit: 是否使用4-bit量化（节省显存）
        """
        self.model_name = model_name or MODEL_NAME
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 初始化 Qwen-VL 模型...")
        print(f"   设备: {self.device}")
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        print(f"\n📥 加载模型: {self.model_name}")
        
        # 配置量化参数 (节省显存)
        if self.load_in_4bit and self.device == "cuda":
            print("   使用 4-bit 量化加载 (节省显存)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            torch_dtype = torch.float16
        else:
            bnb_config = None
            torch_dtype = torch.float32 if self.device == "cpu" else torch.float16
        
        # 加载分词器
        print("   加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # 加载模型
        print("   加载 Model (这可能需要几分钟)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            quantization_config=bnb_config if self.load_in_4bit else None,
            torch_dtype=torch_dtype,
            local_files_only=False,
        ).eval()
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print(f"✓ 模型加载完成！\n")
    
    def chat(self, query: str, image_path: str = None, history: list = None) -> tuple:
        """
        对话接口
        
        Args:
            query: 文本查询
            image_path: 图像路径（可选）
            history: 历史对话记录
            
        Returns:
            (response, history)
        """
        if history is None:
            history = []
        
        # 构建查询
        if image_path and os.path.exists(image_path):
            query_str = f"<img>{image_path}</img>{query}"
        else:
            query_str = query
        
        # 生成回复
        with torch.no_grad():
            response, history = self.model.chat(
                self.tokenizer,
                query=query_str,
                history=history,
                device=self.device
            )
        
        return response, history
    
    def analyze_astronomy_image(self, image_path: str, questions: list = None) -> dict:
        """
        专门用于分析天文图像
        
        Args:
            image_path: 天文图像路径
            questions: 要问的问题列表
            
        Returns:
            分析结果字典
        """
        if not os.path.exists(image_path):
            return {"error": f"图像不存在: {image_path}"}
        
        default_questions = [
            "请描述这张天文图像的内容。",
            "图像中有什么类型的天体？",
            "能看到什么特征？"
        ]
        
        questions = questions or default_questions
        results = {}
        history = []
        
        print(f"🔭 分析天文图像: {image_path}")
        
        for i, q in enumerate(questions):
            print(f"   问题 {i+1}: {q}")
            response, history = self.chat(q, image_path, history)
            results[f"q{i+1}"] = {
                "question": q,
                "answer": response
            }
            print(f"   回答: {response[:100]}...")
        
        return results
    
    def analyze_light_curve_plot(self, plot_path: str) -> str:
        """
        分析光变曲线图
        
        Args:
            plot_path: 光变曲线图像路径
            
        Returns:
            分析文本
        """
        prompt = """这是一张天文光变曲线图。请分析：
1. 这是哪种类型的变星？
2. 周期是多少？
3. 有什么特殊特征？
4. 可能是哪种天体？
请给出专业的天文分析。"""
        
        response, _ = self.chat(prompt, plot_path)
        return response


def interactive_chat(agent: QwenVLAgent):
    """交互式对话模式"""
    print("=" * 60)
    print("🌟 Qwen-VL-8B 交互式对话")
    print("=" * 60)
    print("命令:")
    print("  /img <路径>  - 加载图像")
    print("  /exit        - 退出")
    print("=" * 60)
    
    history = []
    current_image = None
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input == "/exit":
                print("再见！")
                break
            
            if user_input.startswith("/img "):
                img_path = user_input[5:].strip()
                if os.path.exists(img_path):
                    current_image = img_path
                    print(f"✓ 已加载图像: {img_path}")
                else:
                    print(f"✗ 图像不存在: {img_path}")
                continue
            
            # 普通对话
            print("🤖 Qwen: ", end="", flush=True)
            response, history = agent.chat(user_input, current_image, history)
            print(response)
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="Qwen-VL-8B 模型加载器")
    parser.add_argument("--mode", choices=["chat", "analyze"], default="chat",
                       help="运行模式")
    parser.add_argument("--image", type=str, help="图像路径")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                       help="模型名称")
    parser.add_argument("--no-4bit", action="store_true",
                       help="不使用4-bit量化（需要更多显存）")
    
    args = parser.parse_args()
    
    # 初始化模型
    agent = QwenVLAgent(
        model_name=args.model,
        load_in_4bit=not args.no_4bit
    )
    
    if args.mode == "chat":
        interactive_chat(agent)
    elif args.mode == "analyze" and args.image:
        results = agent.analyze_astronomy_image(args.image)
        print("\n分析结果:")
        for key, val in results.items():
            if key.startswith("q"):
                print(f"\nQ: {val['question']}")
                print(f"A: {val['answer']}")
    else:
        print("请提供 --image 参数用于分析模式")


if __name__ == "__main__":
    main()
