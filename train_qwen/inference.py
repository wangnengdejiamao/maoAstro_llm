#!/usr/bin/env python3
"""
AstroSage Qwen 推理脚本
用于测试微调后的模型
"""

import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel


class AstroSageInference:
    """天文助手推理类"""
    
    # 系统提示词
    SYSTEM_PROMPT = """你是 AstroSage，一位专业的天文学专家助手。你精通：
- 恒星演化与赫罗图分析
- 能谱分布(SED)拟合与解释
- 光变曲线分析
- 双星系统与轨道周期
- X射线天文学
- 光谱分析
- 灾变变星(CV)物理

请基于专业知识，准确、详细地回答天文学问题。"""
    
    def __init__(
        self,
        model_path: str,
        use_peft: bool = False,
        base_model_path: str = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.model_path = model_path
        self.use_peft = use_peft
        self.base_model_path = base_model_path
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        print(f"📦 加载模型: {self.model_path}")
        
        # 确定加载哪个分词器
        tokenizer_path = self.model_path
        if self.use_peft and self.base_model_path:
            tokenizer_path = self.base_model_path
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "device_map": "auto",
        }
        
        # 量化
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        
        # 如果是 PEFT 模型
        if self.use_peft and self.base_model_path:
            print(f"   加载基础模型: {self.base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **load_kwargs
            )
            print(f"   加载 LoRA 权重: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        self.model.eval()
        print("✅ 模型加载完成")
    
    def build_prompt(self, question: str, history: List[Dict] = None) -> str:
        """构建 Qwen 格式的 prompt"""
        prompt = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
        
        # 添加历史对话
        if history:
            for turn in history:
                role = turn.get("role", "")
                content = turn.get("content", "")
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # 添加当前问题
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def generate(
        self,
        question: str,
        history: List[Dict] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stream: bool = False,
    ) -> str:
        """生成回答"""
        # 构建 prompt
        prompt = self.build_prompt(question, history)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 生成参数
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # 流式输出
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            generate_kwargs["streamer"] = streamer
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)
        
        # 解码
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def interactive_chat(self):
        """交互式对话"""
        print("\n" + "="*50)
        print("🌟 AstroSage - 天文助手")
        print("="*50)
        print("输入问题开始对话，输入 'quit' 退出\n")
        
        history = []
        
        while True:
            try:
                # 获取输入
                user_input = input("\n👤 用户: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n再见！👋")
                    break
                
                # 生成回答
                print("\n🤖 助手: ", end="", flush=True)
                response = self.generate(
                    user_input,
                    history=history,
                    stream=True
                )
                print()  # 换行
                
                # 更新历史
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                
                # 限制历史长度
                if len(history) > 10:
                    history = history[-10:]
                
            except KeyboardInterrupt:
                print("\n\n再见！👋")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")


def test_questions(model_path: str, use_peft: bool = False, base_model: str = None):
    """测试预设问题"""
    # 测试问题集
    test_questions = [
        "什么是赫罗图？它如何反映恒星的演化？",
        "灾变变星(CV)的光变曲线有什么典型特征？",
        "如何用SED(能谱分布)估计恒星的温度？",
        "双星系统的轨道周期如何测量？",
        "X射线双星有哪些主要类型？",
        "白矮星的光谱有什么特征？",
    ]
    
    print("🧪 运行测试问题...\n")
    
    # 加载模型
    inference = AstroSageInference(
        model_path=model_path,
        use_peft=use_peft,
        base_model_path=base_model
    )
    
    # 测试每个问题
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*50}")
        print(f"问题 {i}/{len(test_questions)}")
        print(f"{'='*50}")
        print(f"Q: {question}")
        print(f"\nA: ", end="", flush=True)
        
        response = inference.generate(question, stream=True)
        print("\n")


def main():
    parser = argparse.ArgumentParser(description="AstroSage Qwen 推理")
    
    parser.add_argument("--model", required=True,
                       help="模型路径 (合并后的模型或LoRA模型)")
    parser.add_argument("--base-model",
                       help="基础模型路径 (使用LoRA时需要)")
    parser.add_argument("--use-peft", action="store_true",
                       help="使用PEFT/LoRA模型")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="4-bit量化加载")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="8-bit量化加载")
    parser.add_argument("--test", action="store_true",
                       help="运行测试问题")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="交互式对话模式")
    parser.add_argument("--question", "-q", type=str,
                       help="单个问题推理")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="采样温度")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="最大生成token数")
    
    args = parser.parse_args()
    
    # 创建推理实例
    inference = AstroSageInference(
        model_path=args.model,
        use_peft=args.use_peft,
        base_model_path=args.base_model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    
    # 运行模式
    if args.test:
        # 测试模式
        test_questions(args.model, args.use_peft, args.base_model)
    elif args.question:
        # 单问题模式
        print(f"Q: {args.question}\n")
        print("A: ", end="", flush=True)
        response = inference.generate(
            args.question,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            stream=True
        )
        print()
    else:
        # 默认交互模式
        inference.interactive_chat()


if __name__ == "__main__":
    main()
