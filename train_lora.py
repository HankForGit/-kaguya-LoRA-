"""
LoRA 微调训练脚本。
基于 HuggingFace PEFT + Trainer，支持 MPS (Apple Silicon)。

用法:
    python train_lora.py
"""
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    ProgressCallback,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments, BatchEncoding,
)

# 确保脚本能找到同目录下的 config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


# ================= 设备选择 =================

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


# ================= 数据处理 =================

def extract_prompt_answer(example: Dict[str, Any]) -> Tuple[str, str]:
    """从多种 JSON 格式中提取 prompt 和 answer。"""
    prompt = ""
    answer = ""

    # 格式 1: messages 列表
    if isinstance(example.get("messages"), list):
        user_msgs = [
            str(m.get("content") or "").strip()
            for m in example["messages"]
            if isinstance(m, dict) and str(m.get("role") or "").strip() == "user"
        ]
        assistant_msgs = [
            str(m.get("content") or "").strip()
            for m in example["messages"]
            if isinstance(m, dict) and str(m.get("role") or "").strip() == "assistant"
        ]
        if user_msgs and assistant_msgs:
            return user_msgs[-1], assistant_msgs[-1]

    # 格式 2: Alpaca (instruction/input/output)
    if "instruction" in example:
        prompt = str(example.get("instruction") or "")
        if example.get("input"):
            prompt = f"{prompt}\n{example['input']}"
        answer = str(example.get("output") or example.get("response") or "")
    elif "q" in example and "a" in example:
        prompt = str(example["q"])
        answer = str(example["a"])
    elif "question" in example and "answer" in example:
        prompt = str(example["question"])
        answer = str(example["answer"])
    elif "prompt" in example and "completion" in example:
        prompt = str(example["prompt"])
        answer = str(example["completion"])

    # 格式 3: 含"用户："和"AI："的纯文本
    if (not prompt or not answer) and isinstance(example.get("text"), str):
        text = example["text"]
        if "用户：" in text and "AI：" in text:
            try:
                user_part, ai_part = text.split("AI：", 1)
                prompt = user_part.replace("用户：", "").strip()
                answer = ai_part.strip()
            except ValueError:
                pass

    return prompt.strip(), answer.strip()


def build_record(tokenizer, example: Dict[str, Any]) -> Dict[str, str]:
    """将一条原始数据转为 {text, prompt, answer} 格式。"""
    # 已有 text 字段的直接用
    if isinstance(example.get("text"), str) and example["text"].strip():
        prompt, answer = extract_prompt_answer(example)
        # 对于已有 text 字段的，用简单启发式找 assistant 起始位置
        return {"text": example["text"].strip(), "prompt": prompt, "answer": answer,
                "prompt_for_mask": ""}

    # messages 格式
    if isinstance(example.get("messages"), list) and example["messages"]:
        cleaned_messages = []
        for item in example["messages"]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role in {"system", "user", "assistant"} and content:
                cleaned_messages.append({"role": role, "content": content})
        if cleaned_messages:
            text = tokenizer.apply_chat_template(
                cleaned_messages, tokenize=False, add_generation_prompt=False
            )
            # 取最后一个 assistant 之前的所有消息作为 prompt 部分
            non_assistant = [m for m in cleaned_messages if m["role"] != "assistant"]
            prompt_for_mask = tokenizer.apply_chat_template(
                non_assistant, tokenize=False, add_generation_prompt=True
            )
            prompt, answer = extract_prompt_answer({"messages": cleaned_messages})
            return {"text": text, "prompt": prompt, "answer": answer,
                    "prompt_for_mask": prompt_for_mask}

    # Alpaca 格式 — instruction 作为 system message，input 作为 user message
    # 如果有 thinking 字段，用 <think> 标签包装
    instruction = str(example.get("instruction") or "").strip()
    user_input = str(example.get("input") or "").strip()
    output = str(example.get("output") or "").strip()
    thinking = str(example.get("thinking") or "").strip()

    if not user_input or not output:
        prompt, answer = extract_prompt_answer(example)
        user_input = user_input or prompt
        output = output or answer

    if user_input and output:
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": user_input})
        assistant_content = f"<think>\n{thinking}\n</think>\n\n{output}" if (cfg.USE_THINKING and thinking) else output
        assistant_messages = list(messages) + [{"role": "assistant", "content": assistant_content}]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                assistant_messages, tokenize=False, add_generation_prompt=False
            )
            # 构造 prompt 部分（不含 assistant），用于后续 mask labels
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"用户：{user_input}\nAI：{output}"
            prompt_text = f"用户：{user_input}\nAI："
        return {"text": text, "prompt": user_input, "answer": output, "prompt_for_mask": prompt_text}

    # 兜底
    parts = [str(v) for v in example.values() if isinstance(v, str) and v.strip()]
    return {"text": "\n".join(parts), "prompt": "", "answer": "", "prompt_for_mask": ""}


def load_datasets(tokenizer) -> Tuple[Dataset, Dataset]:
    """从 config 中指定的多个 JSON 文件加载、合并并拆分为 train/eval。"""
    datasets_list = []
    for file_path in cfg.DATA_FILES:
        file_path = os.path.normpath(os.path.join(cfg.PROJECT_ROOT, file_path))
        if not os.path.exists(file_path):
            print(f"警告: 数据文件 {file_path} 不存在，跳过。")
            continue
        print(f"加载数据: {file_path}")
        raw = load_dataset("json", data_files=file_path, split="train")
        datasets_list.append(raw)

    if not datasets_list:
        raise FileNotFoundError("没有找到任何有效的数据文件，请检查 config.py 中的 DATA_FILES。")

    raw_ds = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    print(f"原始数据共 {len(raw_ds)} 条")

    processed = raw_ds.map(
        lambda x: build_record(tokenizer, x),
        remove_columns=raw_ds.column_names,
    )
    processed = processed.filter(
        lambda x: isinstance(x["text"], str) and x["text"].strip()
    )
    print(f"处理后有效数据: {len(processed)} 条")

    # 按 TRAIN_VAL_SPLIT 拆分训练集和验证集
    split = processed.train_test_split(test_size=1 - cfg.TRAIN_VAL_SPLIT, seed=42)
    print(f"训练集: {len(split['train'])} 条 | 验证集: {len(split['test'])} 条")
    return split["train"], split["test"]


# ================= 自定义回调 =================

class DetailedProgressCallback(TrainerCallback):
    """每步训练后输出进度条，显示 step/epoch/loss/grad_norm/eval_ce/lr。"""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.pbar: Optional[tqdm] = None
        self._last_train_log: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        self.pbar = tqdm(
            total=self.total_steps, desc="训练进度", unit="step",
            dynamic_ncols=True, colour="green",
        )
        return control

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        return control

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> TrainerControl:
        if logs is None:
            return control

        if "loss" in logs and "eval_loss" not in logs:
            self._last_train_log = {
                "loss": float(logs.get("loss", float("nan"))),
                "grad_norm": float(logs.get("grad_norm", float("nan"))),
                "learning_rate": float(logs.get("learning_rate", 0.0)),
                "epoch": float(logs.get("epoch", state.epoch or 0.0)),
            }

        if "eval_loss" in logs:
            merged = {
                "step": float(state.global_step),
                "epoch": float(logs.get("epoch", self._last_train_log.get("epoch", 0.0))),
                "loss": self._last_train_log.get("loss", float("nan")),
                "grad_norm": self._last_train_log.get("grad_norm", float("nan")),
                "eval_cross_entropy": float(logs["eval_loss"]),
                "learning_rate": self._last_train_log.get("learning_rate", 0.0),
            }
            self.history.append(merged)
            self._update_pbar(merged)

        return control

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if state.global_step % args.eval_steps != 0 and self.pbar is not None:
            self.pbar.update(1)
        return control

    def _update_pbar(self, metrics: Dict[str, float]) -> None:
        if self.pbar is None:
            return
        self.pbar.update(1)
        self.pbar.set_postfix({
            "epoch": f"{metrics['epoch']:.2f}",
            "loss": f"{metrics['loss']:.4f}",
            "grad": f"{metrics['grad_norm']:.4f}",
            "eval_ce": f"{metrics['eval_cross_entropy']:.4f}",
            "lr": f"{metrics['learning_rate']:.2e}",
        })
        self.pbar.write(
            f"[step {int(metrics['step']):>5d} | epoch {metrics['epoch']:.2f}] "
            f"loss={metrics['loss']:.4f}  grad_norm={metrics['grad_norm']:.4f}  "
            f"eval_ce={metrics['eval_cross_entropy']:.4f}  lr={metrics['learning_rate']:.2e}"
        )


# ================= 主流程 =================

def main() -> None:
    # ---- 设备 ----
    device = pick_device(cfg.FORCE_MPS)
    print(f"使用设备: {device}")

    if cfg.MPS_FAST_MATH:
        os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ---- Tokenizer ----
    print("加载 tokenizer ...")
    tokenizer = cast(PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(cfg.MODEL_PATH, trust_remote_code=False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- 数据 ----
    train_ds, eval_ds = load_datasets(tokenizer)
    if len(train_ds) == 0:
        raise ValueError("训练集为空，无法训练。")

    # ---- 模型 ----
    print("加载模型 ...")
    model_dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_PATH,
        dtype=model_dtype,
        trust_remote_code=False,
        device_map="auto" if device != "cpu" else None,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ---- LoRA ----
    peft_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- Tokenize ----
    def tokenize_fn(examples: Dict[str, List[str]]) -> BatchEncoding:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.MAX_SEQ_LENGTH,
        )
        # Mask loss on prompt tokens: only compute loss on assistant response
        labels_list = []
        for i in range(len(examples["text"])):
            ids = tokenized["input_ids"][i].copy()
            mask_text = examples.get("prompt_for_mask", [""] * len(examples["text"]))[i]
            if mask_text:
                prompt_ids = tokenizer(mask_text, add_special_tokens=False)["input_ids"]
                mask_len = min(len(prompt_ids), len(ids))
                ids[:mask_len] = [-100] * mask_len
            labels_list.append(ids)
        tokenized["labels"] = labels_list
        return tokenized

    remove_cols = [c for c in train_ds.column_names if c not in ("input_ids", "attention_mask", "labels")]
    tokenized_train = train_ds.map(
        tokenize_fn, batched=True, remove_columns=remove_cols,
    )
    tokenized_eval = eval_ds.map(
        tokenize_fn, batched=True, remove_columns=remove_cols,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---- 训练参数 ----
    effective_batch = cfg.BATCH_SIZE * cfg.GRAD_ACCUM
    steps_per_epoch = (len(tokenized_train) + effective_batch - 1) // effective_batch
    total_steps = steps_per_epoch * cfg.EPOCHS
    print(f"每个 epoch 约 {steps_per_epoch} 步 | 总步数: {total_steps}")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_pin_memory = device == "cuda"

    args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        learning_rate=cfg.LEARNING_RATE,
        num_train_epochs=cfg.EPOCHS,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=cfg.EVAL_STEPS,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=use_pin_memory,
        warmup_steps=cfg.WARMUP_STEPS,
        weight_decay=cfg.WEIGHT_DECAY,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        disable_tqdm=True,          # 由自定义回调接管
        fp16=False,
        bf16=use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    # 替换内置进度回调
    for cb_cls in (ProgressCallback, PrinterCallback):
        try:
            trainer.remove_callback(cb_cls)
        except Exception:
            pass
    detailed_progress = DetailedProgressCallback(total_steps=total_steps)
    trainer.add_callback(detailed_progress)

    # ---- 训练 ----
    print("\n开始训练 ...\n")
    trainer.train()

    # ---- 保存（带时间戳子目录，避免覆盖历史训练结果） ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.OUTPUT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    trainer.model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    print(f"\nLoRA 适配器已保存到: {run_dir}")

    # 评估报告
    last_eval = float("nan")
    step_logs = []
    for item in detailed_progress.history:
        step_logs.append({
            "step": int(item["step"]),
            "epoch": round(item["epoch"], 4),
            "loss": round(item["loss"], 6),
            "grad_norm": round(item["grad_norm"], 6),
            "eval_cross_entropy": round(item["eval_cross_entropy"], 6),
            "learning_rate": round(item["learning_rate"], 10),
        })
        last_eval = item["eval_cross_entropy"]

    report_path = os.path.join(run_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"final_eval_loss": last_eval, "step_logs": step_logs}, f,
                  ensure_ascii=False, indent=2)
    print(f"评估报告已保存: {report_path}")
    print(f"最终 eval_loss: {last_eval:.6f}")

    # 保存运行元信息，方便事后追溯
    run_info = {
        "timestamp": timestamp,
        "model_path": cfg.MODEL_PATH,
        "data_files": cfg.DATA_FILES,
        "use_thinking": cfg.USE_THINKING,
        "lora_r": cfg.LORA_R,
        "lora_alpha": cfg.LORA_ALPHA,
        "epochs": cfg.EPOCHS,
        "learning_rate": cfg.LEARNING_RATE,
        "batch_size": cfg.BATCH_SIZE,
        "grad_accum": cfg.GRAD_ACCUM,
        "max_seq_length": cfg.MAX_SEQ_LENGTH,
        "final_eval_loss": last_eval,
    }
    run_info_path = os.path.join(run_dir, "run_info.json")
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
