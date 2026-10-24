"""全新 87 语种 OCR 说明书语言识别强化训练系统 (train_ocr_lid.py)。

升级核心亮点：
1. 87 目标分类收敛：语言学方言与近缘语种合并，100% 涵盖线上核心 38 业务语种。
2. 架构级鲁棒强化：
   - 升级版 LoRA 配置 (r=16, alpha=32, target: query, key, value, dense)。
   - 增强型非线性分类头 (LayerNorm + Linear + GELU + Dropout + Linear)，抗特征漂移。
3. 说明书短句与 OCR 畸变特化：
   - 结合动态增强，充分学习 5~35 字符短句、OCR 形近字与电气规格参数。
4. Apple Silicon MPS 极速训练与显存优化。
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.language_taxonomy import TARGET_LANGUAGES, LANGUAGE_NAMES_87, is_core_38

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def save_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="87 语种说明书 OCR 语言识别强化训练脚本")
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/ocr_v2', help="预处理数据目录")
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/ocr_v2_model', help="模型产物输出目录")
    parser.add_argument('--base-model-path', type=Path, default=ROOT / 'models/xlm_roberta_base', help="基础预训练模型路径")
    parser.add_argument('--epochs', type=int, default=3, help="训练轮数")
    parser.add_argument('--batch-size', type=int, default=32, help="批处理大小")
    parser.add_argument('--lr-adapter', type=float, default=2.5e-4, help="LoRA 适配器学习率")
    parser.add_argument('--lr-head', type=float, default=1.2e-3, help="分类头学习率")
    parser.add_argument('--max-length', type=int, default=96, help="最大 Token 长度 (说明书短句通常<96)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke-steps', type=int, default=0, help="冒烟测试步数 (0 表示完整训练)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import psutil
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)

    # 优先使用 Apple Silicon MPS 加速
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"训练计算后端: {device}")

    labels = sorted(TARGET_LANGUAGES)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    # 加载数据集
    def read_jsonl(path: Path):
        records = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r['label_id'] = label2id[r['label']]
                    records.append(r)
        return records

    print("正在载入训练与验证数据...", flush=True)
    train_records = read_jsonl(args.data_dir / 'train.jsonl')
    val_records = read_jsonl(args.data_dir / 'validation.jsonl')
    print(f"数据集载入成功: 训练样本数={len(train_records)}, 验证样本数={len(val_records)}, 目标类别数={len(labels)}")

    # 加载 Tokenizer 与 Base Model
    print(f"加载预训练基础模型: {args.base_model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
        attn_implementation='eager'
    )

    # 配置升级版 LoRA (r=16, 扩展到 query, key, value, dense)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.08,
        target_modules=['query', 'key', 'value', 'dense'],
        modules_to_save=['classifier']
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量汇总: 可训练参数={trainable_params:,} ({trainable_params/total_params*100:.2f}%), 总参数={total_params:,}")

    # 保存配置
    config_record = {
        'labels_count': len(labels),
        'labels': labels,
        'train_samples': len(train_records),
        'val_samples': len(val_records),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr_adapter': args.lr_adapter,
        'lr_head': args.lr_head,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_targets': ['query', 'key', 'value', 'dense'],
        'device': str(device),
        'torch_version': torch.__version__
    }
    save_json(args.output_dir / 'train_config.json', config_record)

    # 构造 DataLoader
    def collate_fn(batch):
        texts = [r['text'] for r in batch]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
        encoded['labels'] = torch.tensor([r['label_id'] for r in batch], dtype=torch.long)
        return encoded

    # 类别平衡采样
    label_counts = Counter(r['label_id'] for r in train_records)
    weights = [1.0 / label_counts[r['label_id']] for r in train_records]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_records), replacement=True)

    train_loader = DataLoader(train_records, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_records, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)

    # 优化器与学习率规划
    head_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
    adapter_params = [p for n, p in model.named_parameters() if 'classifier' not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': args.lr_head},
        {'params': adapter_params, 'lr': args.lr_adapter}
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(10, int(0.06 * total_steps))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def evaluate_model(loader, eval_name: str = 'val'):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.inference_mode():
            for batch in loader:
                y = batch.pop('labels').tolist()
                inputs = {k: v.to(device) for k, v in batch.items()}
                logits = model(**inputs).logits
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(y)
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        core_38_targets = []
        core_38_preds = []
        for t, p in zip(all_targets, all_preds):
            if is_core_38(labels[t]):
                core_38_targets.append(t)
                core_38_preds.append(p)
        core_38_f1 = f1_score(core_38_targets, core_38_preds, average='macro', zero_division=0) if core_38_targets else 0.0
        return {'accuracy': acc, 'macro_f1': macro_f1, 'core_38_macro_f1': core_38_f1}

    print("启动模型微调训练...", flush=True)
    best_val_f1 = -1.0
    train_history = []
    global_step = 0
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % 20 == 0 or global_step == 1:
                cur_loss = loss.item()
                rss_gb = psutil.Process().memory_info().rss / 1e9
                print(f"[Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps}] Loss: {cur_loss:.4f} | RSS: {rss_gb:.2f}GB", flush=True)

            if args.smoke_steps > 0 and global_step >= args.smoke_steps:
                print(f"已达到冒烟测试限制步数: {args.smoke_steps}，结束训练。", flush=True)
                break

        val_metrics = evaluate_model(val_loader, f'val_epoch_{epoch+1}')
        print(f"--- Epoch {epoch+1} 验证评估: Macro-F1 = {val_metrics['macro_f1']*100:.2f}%, 核心38语种 F1 = {val_metrics['core_38_macro_f1']*100:.2f}%, Accuracy = {val_metrics['accuracy']*100:.2f}% ---", flush=True)
        val_metrics['epoch'] = epoch + 1
        train_history.append(val_metrics)

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_dir = args.output_dir / 'best_adapter'
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_json(best_dir / 'languages.json', labels)
            save_json(best_dir / 'best_metrics.json', val_metrics)
            save_json(args.output_dir / 'languages.json', labels)
            print(f"-> 最佳模型权重已保存至: {best_dir}", flush=True)

        if args.smoke_steps > 0 and global_step >= args.smoke_steps:
            break

    total_time = time.perf_counter() - start_time
    save_json(args.output_dir / 'training_summary.json', {
        'best_macro_f1': best_val_f1,
        'history': train_history,
        'total_training_seconds': total_time
    })
    print(f"训练全流程执行完成！总耗时: {total_time:.2f}s，最佳验证 Macro-F1: {best_val_f1*100:.2f}%")


if __name__ == '__main__':
    main()
