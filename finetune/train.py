"""
在虛擬機上微調小模型（Qwen3.5-4B，4-bit + LoRA），只學「解析條件」這一步：對話 → JSON。
寫法參考實驗室的 ~/projects/demo-notebooks/fine_tune_qwen3_5.ipynb（unsloth）。

用虛擬機原本的 Python（/opt/venv_torch，裝好了 torch、unsloth；不是跑 API 用的 ~/venvs/foodbot）：
    cd ~/projects/taipei-food-bot
    python finetune/train.py --dry-run          # 只整理資料、看長度，不載入模型、不用 GPU
    python finetune/train.py --steps 300        # 開始訓練（會先檢查 GPU 有沒有足夠的空間）

訓練好的 LoRA 存在 finetune/outputs/<名稱>/（不進 git；環境變數 FINETUNE_OUT 可以改位置）。
中斷後再跑一次，會從最後一個存檔點（每 50 步一個）接著訓練。
"""
import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
# 存檔的地方：預設 finetune/outputs；Colab 上設成 Google Drive，斷線後資料還在
OUT = os.environ.get("FINETUNE_OUT") or os.path.join(HERE, "outputs")
# Qwen3.5 的對話模板：助理回答前面會先放一段空的思考（關閉思考模式時）。訓練只算這之後的 JSON
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION_PART = "<|im_start|>user\n"


def load_rows(path: str) -> list:
    """每一筆的答案都是程式修正後的 JSON（labels.py），老師讀錯的地方也已經改成正確的，全部都拿來訓練。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_text(tokenizer, row: dict) -> str:
    messages = row["messages"] + [{"role": "assistant", "content": row["label"]}]
    return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)


def split_by_dialogue(rows: list, val_ratio: float, seed: int = 42):
    """同一段對話的每一句都放在同一邊（不然驗證集會看到訓練過的前文）。"""
    ids = sorted({r["dialogue_id"] for r in rows})
    random.Random(seed).shuffle(ids)
    val_ids = set(ids[:max(1, int(len(ids) * val_ratio))])
    return [r for r in rows if r["dialogue_id"] not in val_ids], [r for r in rows if r["dialogue_id"] in val_ids]


def gpu_free_gb() -> float:
    import torch
    free, _ = torch.cuda.mem_get_info()
    return free / 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "train.jsonl"))
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--name", default="parser-lora", help="存檔的資料夾名稱")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16, help="LoRA 的大小")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--min-free-gb", type=float, default=10, help="GPU 空間不到這麼多就不開始（GPU 是共用的）")
    ap.add_argument("--dry-run", action="store_true", help="只整理資料，不訓練")
    args = ap.parse_args()

    rows = load_rows(args.data)
    train_rows, val_rows = split_by_dialogue(rows, 0.1)
    print(f"資料：{len(rows)} 筆（訓練 {len(train_rows)}、驗證 {len(val_rows)}）")

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        lengths = sorted(len(tok(to_text(tok, r))["input_ids"]) for r in rows)
        print(f"長度（token）：最短 {lengths[0]}、中位數 {lengths[len(lengths) // 2]}、最長 {lengths[-1]}"
              f"（上限 {args.max_len}，超過的 {sum(n > args.max_len for n in lengths)} 筆）")
        sample = to_text(tok, rows[0])
        print("範例（結尾）：\n" + sample[-600:])
        assert RESPONSE_PART in sample, "對話模板跟預期的不一樣，RESPONSE_PART 要調整"
        return

    free = gpu_free_gb()
    if free < args.min_free_gb:
        raise SystemExit(f"GPU 只剩 {free:.1f} GB（需要 {args.min_free_gb} GB），可能有人在用，晚點再跑")

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_len, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407, max_seq_length=args.max_len)

    def dataset(rs):
        return Dataset.from_list([{"text": to_text(tokenizer, r)} for r in rs])

    out_dir = os.path.join(OUT, args.name)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset(train_rows), eval_dataset=dataset(val_rows),
        args=SFTConfig(
            output_dir=out_dir + "-checkpoints",
            dataset_text_field="text", dataset_num_proc=1, max_seq_length=args.max_len,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            warmup_ratio=0.05, max_steps=args.steps, learning_rate=args.lr,
            lr_scheduler_type="cosine", weight_decay=0.01, optim="adamw_8bit",
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=10, eval_strategy="steps", eval_steps=50,
            save_strategy="steps", save_steps=50, save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            report_to="none", seed=3407,
        ),
    )
    # 只訓練助理的回答（JSON），不訓練系統提示詞和使用者的話
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)
    # 上次斷在一半：從最後一個存檔點接著訓練
    ckpt_dir = out_dir + "-checkpoints"
    resume = os.path.isdir(ckpt_dir) and any(d.startswith("checkpoint-") for d in os.listdir(ckpt_dir))
    if resume:
        print(f"從 {ckpt_dir} 的存檔點接著訓練")
    trainer.train(resume_from_checkpoint=resume or None)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "train_rows": len(train_rows), "val_rows": len(val_rows)}, f,
                  ensure_ascii=False, indent=2)
    print(f"已存到 {out_dir}")


if __name__ == "__main__":
    main()
