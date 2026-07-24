"""
LoRA 模型推理脚本。
加载基座模型 + LoRA 适配器，合并后进行交互式对话。

用法:
    python infer_lora.py [--lora_path ./lora_output]
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, cast

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def find_latest_lora(base_dir: str) -> str | None:
    """在 base_dir 下找最新的时间戳子目录，返回路径。找不到则返回 None。"""
    if not os.path.isdir(base_dir):
        return None
    subdirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and os.path.exists(os.path.join(base_dir, d, "adapter_config.json"))
    ]
    if not subdirs:
        return None
    # 按时间戳字符串排序，取最新
    subdirs.sort(reverse=True)
    return os.path.join(base_dir, subdirs[0])


def pick_device(force_mps: bool = False) -> str:
    if force_mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("已设置 FORCE_MPS=True，但当前环境不可用 MPS。")
        return "mps"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_prompt(tokenizer, user_text: str, system_prompt: str = "") -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return f"用户：{user_text}\nAI："


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 模型推理")
    parser.add_argument(
        "--lora_path", type=str, default=cfg.OUTPUT_DIR,
        help="LoRA 适配器路径"
    )
    parser.add_argument(
        "--system-prompt", type=str, default="",
        help="可选的 system prompt"
    )
    args = parser.parse_args()

    lora_path = os.path.normpath(os.path.join(os.path.dirname(__file__), args.lora_path))

    # 如果传入的是基础目录（没有 adapter_config.json），自动找最新的时间戳子目录
    if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        latest = find_latest_lora(lora_path)
        if latest:
            print(f"自动选择最新训练结果: {latest}")
            lora_path = latest

    if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        print(f"错误: LoRA 适配器路径不存在或缺少 adapter_config.json: {lora_path}")
        print("请先运行 train_lora.py 训练，或使用 --lora_path 指定路径。")
        sys.exit(1)

    device = pick_device(cfg.FORCE_MPS)

    # ---- 运行摘要 ----
    run_info_path = os.path.join(lora_path, "run_info.json")
    run_info = {}
    if os.path.exists(run_info_path):
        with open(run_info_path) as f:
            run_info = json.load(f)
    print("=" * 60)
    print(f"  基座模型:     {cfg.MODEL_PATH}")
    print(f"  LoRA 适配器:  {lora_path}")
    if run_info:
        print(f"  训练时间:     {run_info.get('timestamp', '?')}")
        print(f"  使用思考链:   {run_info.get('use_thinking', '?')}")
        print(f"  训练数据:     {run_info.get('data_files', '?')}")
        eval_loss = run_info.get('final_eval_loss')
        if eval_loss is not None:
            print(f"  最终 eval:    {eval_loss:.4f}")
    else:
        print("  (无 run_info.json，可能为旧版训练结果)")
    print("=" * 60)

    print(f"使用设备: {device}")

    if cfg.MPS_FAST_MATH:
        os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ---- Tokenizer ----
    tokenizer = cast(PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(cfg.MODEL_PATH, trust_remote_code=False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- 基座模型 ----
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    device_map = "auto" if device != "cpu" else None
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_PATH,
            dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            device_map=device_map,
            attn_implementation="sdpa",
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_PATH,
            dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

    # ---- 加载并合并 LoRA ----
    model = PeftModel.from_pretrained(base_model, lora_path)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.eval()

    print(f"模型加载完成，LoRA 已合并。")
    if device == "mps" and cfg.MPS_GREEDY_DECODING:
        print("MPS 快速模式：使用 greedy decoding。")

    # ---- 交互推理 ----
    print("输入内容开始对话，输入空行退出。\n")
    while True:
        try:
            user_text = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_text:
            break

        prompt = build_prompt(tokenizer, user_text, args.system_prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        input_len = int(inputs["input_ids"].shape[-1])
        do_sample = not (device == "mps" and cfg.MPS_GREEDY_DECODING)
        if device == "mps":
            torch.mps.synchronize()

        # 构建停止 token 列表：正常的 eos + <|endoftext|>（Qwen3.5 有时会生成它）
        stop_token_ids = [tokenizer.eos_token_id]
        endoftext_tokens = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
        if endoftext_tokens:
            stop_token_ids.append(endoftext_tokens[0])

        gen_kwargs: Dict = dict(
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            do_sample=do_sample,
            num_beams=1,
            use_cache=True,
            repetition_penalty=cfg.REPETITION_PENALTY,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = cfg.TEMPERATURE
            gen_kwargs["top_p"] = cfg.TOP_P

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)

        if device == "mps":
            torch.mps.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-6)
        gen_len = max(int(out[0].shape[-1]) - input_len, 0)
        print(f"[{gen_len / elapsed:.1f} tok/s]")

        gen_ids = out[0][input_len:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=False)
        # Strip special tokens and thinking content for display
        import re
        answer = answer.replace("<|im_start|>", "").replace("<|im_end|>", "")
        answer = answer.replace("<|endoftext|>", "")
        answer = re.sub(r"<think>.*?</think>\n*", "", answer, flags=re.DOTALL)
        # Clean up role prefixes
        answer = re.sub(r"(system|user|assistant)\n?", "", answer)
        answer = answer.strip()
        print(f"AI: {answer}\n")


if __name__ == "__main__":
    main()
