from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 你的模型路径
model_name_or_path = "/root/autodl-fs/models/baichuan-7b"

try:
    # 加载 tokenizer
    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    # 加载模型
    print("正在加载 Baichuan-7B 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ 模型加载成功！")

    # 简单测试推理
    print("\n开始测试生成...")
    inputs = tokenizer("你好，请介绍一下你自己", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🤖 模型回答：", response)

except Exception as e:
    print("\n❌ 加载失败，错误信息：")
    print(e)