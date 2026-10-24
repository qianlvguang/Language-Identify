"""四阶梯全方位鲁棒性评估系统 (evaluate_robustness.py)。

用于针对说明书 OCR 实际业务环境的多维度难度分级评估：
1. Level 1: Standard In-Domain (标准长句，无噪声)
2. Level 2: Short Text (<=35 字符极短文本与词组)
3. Level 3: Light OCR Noise (形近字、错别字与连词)
4. Level 4: Heavy OCR & Specs (重度乱码、变音丢失与说明书参数混合)

输出指标：
- 87 类全局 Macro-F1 与 Accuracy
- 线上核心 38 业务语种专属 Macro-F1
- 逐语言指标 CSV 与最易混淆语言对分析
"""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.language_taxonomy import TARGET_LANGUAGES, LANGUAGE_NAMES_87, is_core_38, ONLINE_CORE_38

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def main():
    parser = argparse.ArgumentParser(description="87 语种说明书 OCR 语言识别四阶梯鲁棒性评测")
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/ocr_v2', help="测试集所在目录")
    parser.add_argument('--adapter-path', type=Path, required=True, help="待评测的 LoRA 适配器路径")
    parser.add_argument('--base-model-path', type=Path, default=ROOT / 'models/xlm_roberta_base', help="基础模型路径")
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/robustness_v2', help="评测报告产出目录")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=96)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"评估运行后端: {device}")

    labels = sorted(TARGET_LANGUAGES)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    # 加载 Tokenizer 与 Model
    print(f"载入模型及适配器: {args.adapter_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
        attn_implementation='eager'
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.to(device)
    model.eval()

    def load_jsonl(path: Path):
        records = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r['label_id'] = label2id[r['label']]
                    records.append(r)
        return records

    levels = [
        ('level1_standard', args.data_dir / 'test_standard.jsonl', '标准原文基准'),
        ('level2_short', args.data_dir / 'test_short.jsonl', '短句测试 (<=35字符)'),
        ('level3_light_ocr', args.data_dir / 'test_light_ocr.jsonl', '轻度 OCR 错字'),
        ('level4_heavy_ocr', args.data_dir / 'test_heavy_ocr.jsonl', '重度 OCR 乱码与规格参数')
    ]

    all_level_summaries = {}
    detailed_per_lang = defaultdict(dict)

    for level_key, test_path, level_desc in levels:
        if not test_path.exists():
            print(f"跳过 {level_desc}: 文件未找到 ({test_path})")
            continue

        print(f"正在评测: {level_desc} ({test_path.name})...", flush=True)
        records = load_jsonl(test_path)

        def collate_fn(batch):
            texts = [r['text'] for r in batch]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
            encoded['labels'] = torch.tensor([r['label_id'] for r in batch], dtype=torch.long)
            return encoded

        loader = DataLoader(records, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        all_preds = []
        all_targets = []
        start_t = time.perf_counter()

        with torch.inference_mode():
            for batch in loader:
                y = batch.pop('labels').tolist()
                inputs = {k: v.to(device) for k, v in batch.items()}
                logits = model(**inputs).logits
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(y)

        elapsed = time.perf_counter() - start_t
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        # 计算核心 38 语种指标
        c38_targets, c38_preds = [], []
        for t, p in zip(all_targets, all_preds):
            if is_core_38(labels[t]):
                c38_targets.append(t)
                c38_preds.append(p)
        c38_f1 = f1_score(c38_targets, c38_preds, average='macro', zero_division=0) if c38_targets else 0.0

        # 分类详细报告
        cls_rep = classification_report(all_targets, all_preds, target_names=labels, output_dict=True, zero_division=0)
        for l in labels:
            detailed_per_lang[l][level_key] = cls_rep[l]['f1-score']

        level_summary = {
            'level': level_key,
            'description': level_desc,
            'total_samples': len(records),
            'accuracy': acc,
            'macro_f1': macro_f1,
            'core_38_macro_f1': c38_f1,
            'latency_per_sample_ms': (elapsed / len(records)) * 1000
        }
        all_level_summaries[level_key] = level_summary
        print(f"-> {level_desc}: Macro-F1 = {macro_f1*100:.2f}%, 核心38语种 F1 = {c38_f1*100:.2f}%, 准确率 = {acc*100:.2f}%, 耗时: {elapsed:.2f}s")

    # 汇总保存
    summary_path = args.output_dir / 'robustness_summary.json'
    summary_path.write_text(json.dumps(all_level_summaries, ensure_ascii=False, indent=2), encoding='utf-8')

    # 生成逐语言 CSV
    csv_path = args.output_dir / 'per_language_robustness.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        header = ['code', 'name', 'is_core_38', 'standard_f1', 'short_f1', 'light_ocr_f1', 'heavy_ocr_f1']
        writer.writerow(header)
        for l in labels:
            row = [
                l,
                LANGUAGE_NAMES_87.get(l, l),
                'YES' if is_core_38(l) else 'NO',
                f"{detailed_per_lang[l].get('level1_standard', 0.0)*100:.2f}%",
                f"{detailed_per_lang[l].get('level2_short', 0.0)*100:.2f}%",
                f"{detailed_per_lang[l].get('level3_light_ocr', 0.0)*100:.2f}%",
                f"{detailed_per_lang[l].get('level4_heavy_ocr', 0.0)*100:.2f}%"
            ]
            writer.writerow(row)

    print(f"评估报告生成完成！已产出至: {args.output_dir}")


if __name__ == '__main__':
    main()
