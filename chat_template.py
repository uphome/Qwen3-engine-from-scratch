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


def format_chat(messages: list[dict]) -> str:
    """
    将 messages 列表转换为 Qwen3 chat template 字符串

    Args:
        messages: [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."}, ...]

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
    return text
