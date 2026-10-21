"""在 Apple MPS 上训练并评估广语言覆盖的 XLM-R LoRA 分类器。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HOME', str(ROOT / 'data/hf_cache'))

# 使用数据集的 ISO 639-3 标签；不把语言和书写系统混为一类。
LANGUAGES = '''afr amh ara asm aze bel ben bod bos bre bul cat ces cym dan deu ell eng
epo est eus fas fin fra fry gle glg guj hau heb hin hrv hun hye ibo ind isl ita jav
jpn kan kat kaz khm kin kir kor kur lao lat lav lit ltz mal mar mkd mlg mon mri
msa mlt mya nep nld nno nob ori orm pan pol por pus ron rus san sin slk slv snd som
sqi srp sun swa swe tam tel tgk tgl tha tuk tur uig ukr urd uzb vie xho yid yor zho
fao ckb hat sna tat ceb spa gla'''.split()

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def prepare(args):
    import pyarrow.parquet as pq
    meta = json.loads((ROOT / 'data/wili_metadata.json').read_text())
    names = meta['cardData']['dataset_info']['features'][1]['dtype']['class_label']['names']
    labels = sorted(LANGUAGES)
    assert len(labels) == len(set(labels)) and 'isl' in labels
    assert set('eng zho spa ara hin ben por rus jpn deu fra ita kor tur vie tha ind urd isl'.split()) <= set(labels)
    assert not (set(labels) - set(names.values()))
    mapping = {code: i for i, code in enumerate(labels)}
    # 测试集优先去重，训练和验证中删除测试集完整文本的重复项。
    seen = set()
    buckets = {}
    for split in ['test', 'train']:
        groups = {code: [] for code in labels}
        source = ROOT / 'data' / f'{split}.parquet'
        embedded = json.loads(pq.read_schema(source).metadata[b'huggingface'])
        assert embedded['info']['features']['label']['names'] == [names[str(i)] for i in range(len(names))]
        table = pq.read_table(source).to_pylist()
        for row in table:
            code = names[str(row['label'])]
            if code not in mapping:
                continue
            text = unicodedata.normalize('NFC', row['sentence']).strip()
            normalized = ' '.join(text.split()).casefold()
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            if digest in seen or sum(c.isalpha() for c in text) < 10:
                continue
            seen.add(digest)
            groups[code].append({'text': text, 'label': mapping[code], 'hash': digest})
        buckets[split] = groups
    result = {'train': [], 'validation': [], 'test': []}
    counts = {}
    rng = random.Random(args.seed)
    for code in labels:
        train = buckets['train'][code]
        test = buckets['test'][code]
        rng.shuffle(train)
        rng.shuffle(test)
        n_val = min(args.val_per_language, max(1, len(train) // 5))
        n_train = min(args.train_per_language, len(train) - n_val)
        n_test = min(args.test_per_language, len(test))
        if n_train < 20 or n_val < 5 or n_test < 20:
            raise ValueError(f'{code}: 有效样本不足以评估 train={len(train)}, test={len(test)}')
        result['train'] += train[:n_train]
        result['validation'] += train[n_train:n_train + n_val]
        result['test'] += test[:n_test]
        counts[code] = {'available_train': len(train), 'available_test': len(test),
                        'train': n_train, 'validation': n_val, 'test': n_test}
    for split, rows in result.items():
        rng.shuffle(rows)
        with (args.output / f'{split}_manifest.jsonl').open('w') as f:
            for row in rows:
                f.write(json.dumps({'hash': row['hash'], 'label': labels[row['label']]}) + '\n')
    save_json(args.output / 'languages.json', labels)
    save_json(args.output / 'data_counts.json', counts)
    print(json.dumps({'data_samples': {k: len(v) for k, v in result.items()},
                      'languages_below_requested_counts': {k: v for k, v in counts.items()
                       if v['train'] < args.train_per_language or v['validation'] < args.val_per_language
                       or v['test'] < args.test_per_language}}, ensure_ascii=False), flush=True)
    return result, labels

def load_prepared(args):
    labels = sorted(LANGUAGES)
    mapping = {s: i for i, s in enumerate(labels)}
    result = {}
    counts = {s: {} for s in labels}
    fingerprints = {}
    for split in ['train', 'validation', 'test']:
        path = args.prepared_data / f'{split}.jsonl'
        records = [json.loads(s) for s in path.read_text().splitlines()]
        fingerprints[split] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        for record in records:
            record['label'] = mapping[record['label']]
        result[split] = records
        for label in labels:
            counts[label][split] = sum(r['label'] == mapping[label] for r in records)
        with (args.output / f'{split}_manifest.jsonl').open('w') as f:
            for r in records:
                f.write(json.dumps({k: v for k, v in r.items() if k != 'text'}) + '\n')
    assert {r['label'] for r in result['train']} == set(range(len(labels)))
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not ({r['hash'] for r in result[a]} & {r['hash'] for r in result[b]})
        assert not ({r['parent'] for r in result[a]} & {r['parent'] for r in result[b]})
    save_json(args.output / 'languages.json', labels)
    save_json(args.output / 'data_counts.json', counts)
    save_json(args.output / 'prepared_fingerprints.json', fingerprints)
    return result, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/local_lora_109')
    parser.add_argument('--train-per-language', type=int, default=200)
    parser.add_argument('--val-per-language', type=int, default=50)
    parser.add_argument('--test-per-language', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--prepared-data', type=Path)
    parser.add_argument('--init-adapter', type=Path)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import psutil
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS 不可用：停止，避免悄悄退回 CPU 冒充本地 GPU 训练。')
    device = torch.device('mps')
    host_before = {'available_gb': psutil.virtual_memory().available / 1e9,
                   'swap_used_gb': psutil.swap_memory().used / 1e9}
    data, labels = load_prepared(args) if args.prepared_data else prepare(args)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/xlm_roberta_base', local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROOT / 'models/xlm_roberta_base', num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={x: i for i, x in enumerate(labels)},
        local_files_only=True, attn_implementation='eager')
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['query', 'value'], modules_to_save=['classifier']))
    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        weights = load_file(str(args.init_adapter / 'adapter_model.safetensors'))
        set_peft_model_state_dict(model, weights)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'output': str(args.output), 'languages': len(labels),
              'device': str(device), 'torch': torch.__version__, 'trainable': trainable,
              'total': total, 'precision': 'float32', 'lora_r': 8, 'lora_alpha': 16,
              'learning_rate': 2e-4, 'classifier_learning_rate': 1e-3,
              'dataset': 'OPUS-100 + filtered WiLI + training-only augmentation' if args.prepared_data else 'MartinThoma/wili_2018',
              'scope': 'source-separated sentence test' if args.prepared_data else 'Wikipedia in-domain pilot'}
    config['host_before_model'] = host_before
    config['sampling'] = 'uniform_language_weighted_with_replacement' if args.prepared_data else 'shuffle'
    config['base_revision'] = json.loads((ROOT / 'models/base_metadata.json').read_text())['sha']
    if args.init_adapter:
        config['initial_adapter_sha256'] = hashlib.file_digest(
            (args.init_adapter / 'adapter_model.safetensors').open('rb'), 'sha256').hexdigest()
    save_json(args.output / 'config.json', config)
    print(json.dumps(config, ensure_ascii=False), flush=True)

    def make_loader(rows, shuffle=False, short=False):
        def collate(batch):
            texts = [r['text'][:40] if short else r['text'] for r in batch]
            encoded = tokenizer(texts, padding=True, truncation=True,
                                max_length=args.max_length, return_tensors='pt')
            encoded['labels'] = torch.tensor([r['label'] for r in batch])
            return encoded
        sampler = None
        if shuffle and args.prepared_data:
            from torch.utils.data import WeightedRandomSampler
            frequencies = Counter(r['label'] for r in rows)
            sampler = WeightedRandomSampler([1. / frequencies[r['label']] for r in rows],
                                            num_samples=len(rows), replacement=True)
        return DataLoader(rows, batch_size=args.batch_size, shuffle=shuffle and sampler is None,
                          sampler=sampler, num_workers=0, collate_fn=collate)

    def evaluate(rows, name, short=False):
        model.eval()
        truth, preds, confidences = [], [], []
        started = time.perf_counter()
        with torch.inference_mode():
            for batch in make_loader(rows, short=short):
                truth.extend(batch.pop('labels').tolist())
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                prob = logits.softmax(-1).cpu()
                preds.extend(prob.argmax(-1).tolist())
                confidences.extend(prob.max(-1).values.tolist())
        report = classification_report(truth, preds, labels=list(range(len(labels))),
                                       target_names=labels, output_dict=True, zero_division=0)
        evaluated_labels = sorted(set(truth))
        summary = {'samples': len(truth), 'accuracy': accuracy_score(truth, preds),
                   'evaluated_languages': len(evaluated_labels),
                   'missing_languages': [s for i, s in enumerate(labels) if i not in evaluated_labels],
                   'macro_f1': f1_score(truth, preds, labels=evaluated_labels,
                                        average='macro', zero_division=0),
                   'icelandic': report['isl'], 'seconds': time.perf_counter() - started}
        save_json(args.output / f'{name}_summary.json', summary)
        save_json(args.output / f'{name}_per_language.json', report)
        matrix = confusion_matrix(truth, preds, labels=list(range(len(labels))))
        with (args.output / f'{name}_confusion.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['true/predicted'] + labels)
            for label, row in zip(labels, matrix.tolist()):
                writer.writerow([label] + row)
        with (args.output / f'{name}_predictions.jsonl').open('w') as f:
            for row, y, pred, conf in zip(rows, truth, preds, confidences):
                f.write(json.dumps({'hash': row['hash'], 'true': labels[y],
                                    'predicted': labels[pred], 'confidence': conf}) + '\n')
        print(json.dumps({'evaluation': name, **summary}), flush=True)
        return summary

    if args.evaluate_only:
        if not args.init_adapter:
            raise ValueError('评测旧模型必须明确指定 --init-adapter')
        test = evaluate(data['test'], 'test')
        short = evaluate(data['test'], 'test_first_40_characters', short=True)
        save_json(args.output / 'result.json', {'test': test, 'short_text': short})
        return

    head, adapters = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (head if 'classifier' in name else adapters).append(p)
    optimizer = torch.optim.AdamW([{'params': head, 'lr': 1e-3},
                                   {'params': adapters, 'lr': 2e-4}], weight_decay=0.01)
    train_loader = make_loader(data['train'], shuffle=True)
    all_steps = len(train_loader) * args.epochs
    warmup = max(1, int(0.06 * all_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup,
                                   max(0., (all_steps - step) / max(1, all_steps - warmup))))
    torch.mps.synchronize()
    started = time.perf_counter()
    step = 0
    best = -1
    history = []
    with (args.output / 'training.jsonl').open('w', buffering=1) as log:
        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(**{k: v.to(device) for k, v in batch.items()})
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                step += 1
                if step == 1 or step % 10 == 0:
                    torch.mps.synchronize()
                    record = {'epoch': epoch + 1, 'step': step, 'total_steps': all_steps,
                              'loss': float(output.loss.detach().cpu()),
                              'elapsed_seconds': time.perf_counter() - started,
                              'rss_gb': psutil.Process().memory_info().rss / 1e9,
                              'system_available_gb': psutil.virtual_memory().available / 1e9,
                              'swap_used_gb': psutil.swap_memory().used / 1e9,
                              'mps_allocated_gb': torch.mps.current_allocated_memory() / 1e9,
                              'mps_driver_gb': torch.mps.driver_allocated_memory() / 1e9}
                    log.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
                if args.smoke_steps and step >= args.smoke_steps:
                    save_json(args.output / 'smoke_result.json', record)
                    return
            summary = evaluate(data['validation'], f'validation_epoch_{epoch + 1}')
            history.append(summary)
            if summary['macro_f1'] > best:
                best = summary['macro_f1']
                model.save_pretrained(args.output / 'best_adapter')
            model.train()
    training_seconds = time.perf_counter() - started
    # 按验证集选取 checkpoint，测试集只在选择结束后使用。
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    weights = load_file(str(args.output / 'best_adapter/adapter_model.safetensors'))
    set_peft_model_state_dict(model, weights)
    tokenizer.save_pretrained(args.output / 'best_adapter')
    test = evaluate(data['test'], 'test')
    short = evaluate(data['test'], 'test_first_40_characters', short=True)
    save_json(args.output / 'result.json', {'validation_history': history, 'test': test,
              'short_text': short, 'training_and_validation_seconds': training_seconds,
              'total_seconds': time.perf_counter() - started,
              'limitations': ['未验证实际业务效果', '未做近重复文档聚类',
                              '40字符截断为派生测试，非独立真实短文本集',
                              '未校准置信度，未验证未知语言和混合语言拒识',
                              '测试文本可能出现在基座预训练语料中']} )
    print('实验完成：' + str(args.output / 'result.json'), flush=True)

if __name__ == '__main__':
    main()
