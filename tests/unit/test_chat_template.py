"""Chat template：Qwen3 特殊 token 格式与 thinking 开关"""

from chat_template import format_chat


def test_system_user_assistant_sequence():
    text = format_chat([
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
    ], enable_thinking=True)

    assert text.startswith("<|im_start|>system\n你是助手<|im_end|>\n")
    assert "<|im_start|>user\n你好<|im_end|>\n" in text
    assert text.endswith("<|im_start|>assistant\n")
    assert "<think" not in text


def test_thinking_disabled_appends_empty_think_block():
    text = format_chat([{"role": "user", "content": "hi"}], enable_thinking=False)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_multi_turn_history():
    text = format_chat([
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "再加 1"},
    ])
    assert text.count("<|im_start|>user") == 2
    assert text.count("<|im_start|>assistant") == 2   # 历史中 1 个 + 结尾 1 个
    assert "<|im_start|>assistant\n2<|im_end|>" in text


def test_empty_messages_only_assistant_header():
    text = format_chat([], enable_thinking=True)
    assert text == "<|im_start|>assistant\n"
