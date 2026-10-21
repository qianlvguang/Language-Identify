"""读取本地基座与适配器，输出语言模型分数（未经置信度校准）。"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'data/hf_cache'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text', nargs='+')
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/expanded_lora_109')
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    labels = json.loads((args.run / 'languages.json').read_text())
    config = json.loads((args.run / 'config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/xlm_roberta_base', local_files_only=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        ROOT / 'models/xlm_roberta_base', num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={s: i for i, s in enumerate(labels)},
        local_files_only=True)
    model = PeftModel.from_pretrained(base, args.run / 'best_adapter', local_files_only=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device).eval()
    with torch.inference_mode():
        for text in args.text:
            batch = tokenizer(text, return_tensors='pt', truncation=True,
                              max_length=config['max_length'])
            values, indices = model(**{k: v.to(device) for k, v in batch.items()}).logits.softmax(-1).cpu().topk(3)
            print(json.dumps({'text': text, 'top3': [
                {'language': labels[i], 'score': float(p)}
                for p, i in zip(values[0], indices[0])]}, ensure_ascii=False))

if __name__ == '__main__':
    main()
