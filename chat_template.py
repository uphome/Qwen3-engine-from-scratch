"""
Qwen3 Chat Template — 手动实现 Jinja 模板逻辑

将 messages 列表格式化为 Qwen3 的 chat format:
  <|im_start|>system
  {content}<|im_end|>
  <|im_start|>user
  {content}<|im_end|>
  <|im_start|>assistant
  {content}<|im_end|>
"""


def format_chat(messages: list[dict], enable_thinking: bool = True) -> str:
    """
    将 messages 列表转换为 Qwen3 chat template 字符串

    Args:
        messages: [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."}, ...]
        enable_thinking: True=思考模式开启（跟官方模板默认一致，模型自己输出
            思考链）；False=关闭，在 assistant 后追加一个"空思考块"
            （' thinking\\n\\n response\\n\\n'）——模型训练识别此为关闭信号，
            直接回答。对齐官方 tokenizer_config.json 的渲染逻辑。

    Returns:
        格式化后的字符串，以 "<|im_start|>assistant\\n" 结尾
    """
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    if not enable_thinking:
        # 与 Qwen3 官方模板逐字符一致：空 think 块 = 思考关闭信号
        text += " thinking\n\n response\n\n"
    return text
