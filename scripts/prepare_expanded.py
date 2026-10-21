"""扩充真实短句，独立筛查训练噪声，并仅对训练数据构造变体。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import time
import unicodedata
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]

def norm(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())

def fingerprint(text):
    return hashlib.sha256(norm(text).casefold().encode()).hexdigest()

def page(job):
    config, split, offset = job
    path = ROOT / 'data/opus_pages' / config / f'{split}_{offset}.json'
    headers = path.with_suffix('.headers')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not headers.exists():
        url = 'https://datasets-server.huggingface.co/rows?' + urlencode({
            'dataset': 'Helsinki-NLP/opus-100', 'config': config, 'split': split,
            'offset': offset, 'length': 100})
        for retry in range(4):
            result = subprocess.run(['curl', '-fsSL', '--max-time', '30', '-D', str(headers), url], capture_output=True)
            if result.returncode == 0:
                try:
                    payload = json.loads(result.stdout)
                    assert 'rows' in payload
                    path.write_bytes(result.stdout)
                    break
                except (ValueError, AssertionError):
                    pass
            time.sleep(2 * (retry + 1))
        else:
            path.with_suffix('.error.txt').write_text(result.stderr.decode(errors='replace')[:2000])
            return job, None
    return job, json.loads(path.read_text())

def main():
    import fasttext
    import pycountry
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from local_experiment import LANGUAGES
    from read_opus_files import page as file_page
    labels = sorted(LANGUAGES)
    meta = json.loads((ROOT / 'data/opus_metadata.json').read_text())
    configs = [x for x in meta['cardData']['dataset_info'] if 'en' in x['config_name'].split('-')]
    code_map = {}
    for code in labels:
        lang = pycountry.languages.get(alpha_3=code)
        if lang and hasattr(lang, 'alpha_2'):
            code_map[lang.alpha_2] = code
    rng = random.Random(20261021)
    jobs = []
    pairs = {}
    for entry in sorted(configs, key=lambda x: x['config_name']):
        config = entry['config_name']
        other = next(x for x in config.split('-') if x != 'en')
        if other not in code_map:
            continue
        splits = {x['name']: x['num_examples'] for x in entry['splits']}
        if 'train' not in splits:
            continue
        pairs[config] = code_map[other]
        jobs += [(config, split, 0) for split in ['train', 'validation', 'test'] if split in splits]
    print(json.dumps({'language_pairs': len(pairs), 'requests': len(jobs),
                      'missing_opus_languages': sorted(set(labels) - set(pairs.values()) - {'eng'})}), flush=True)
    payloads = {}
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(file_page, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            job, payload = future.result()
            if payload is None:
                failures.append(job)
            else:
                payloads[job] = payload
            if i % 25 == 0:
                print(json.dumps({'downloaded_pages': i, 'total': len(jobs), 'failures': len(failures)}), flush=True)
    if failures:
        (ROOT / 'data/opus_download_failures.json').write_text(json.dumps(failures))
        raise RuntimeError(f'{len(failures)} 个页面失败，保留缓存后可重跑，不静默缩减语言覆盖')

    revisions = {payload['revision'] for payload in payloads.values()}
    if revisions != {meta['sha']}:
        raise RuntimeError(f'API 版本与元信息不一致：{revisions}')
    teacher = fasttext.load_model(str(ROOT / 'models/lid.176.ftz'))
    teacher_labels = {s.replace('__label__', '') for s in teacher.get_labels()}
    reverse = {v: k for k, v in code_map.items()}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/xlm_roberta_base', local_files_only=True)
    baseline_seen = set()
    for split in ['train', 'validation', 'test']:
        manifest = ROOT / 'runs/local_lora_109' / f'{split}_manifest.jsonl'
        baseline_seen.update(json.loads(s)['hash'] for s in manifest.read_text().splitlines())
    baseline_token_inputs = set()
    for split in ['train', 'test']:
        for item in pq.read_table(ROOT / 'data' / f'{split}.parquet', columns=['sentence']).to_pylist():
            text = norm(item['sentence'])
            if fingerprint(text) in baseline_seen:
                ids = tokenizer.encode(text, truncation=True, max_length=96)
                baseline_token_inputs.add(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    stats = Counter()
    def usable(text):
        return sum(c.isalpha() for c in text) >= 10 and len(text) <= 2000
    def consistent(text, label):
        target = reverse.get(label)
        if target not in teacher_labels:
            return True
        # 使用低层接口，避免旧 FastText predict 与 NumPy 2 的 copy=False 兼容问题。
        prediction = teacher.f.predict(norm(text) + '\n', 1, 0., 'strict')
        if not prediction:
            return False
        score, predicted = prediction[0]
        predicted = predicted.replace('__label__', '')
        if predicted == target:
            return True
        # 教师不能可靠区分这些近缘类别，保留来源标签，不用教师强行重标。
        families = [{'bs', 'hr', 'sr', 'sh'}, {'su', 'jv', 'id', 'ms'}, {'ht', 'fr'}]
        if any(target in family and predicted in family for family in families):
            return True
        # 保留不确定项，只删除明确的跨语言冲突；例如非英语标签下的整段英文。
        return score < .75
    def row(text, label, source, parent):
        text = norm(text)
        return {'text': text, 'label': label, 'hash': fingerprint(text),
                'source': source, 'parent': parent, 'synthetic': False}
    candidates = {s: defaultdict(list) for s in ['train', 'validation', 'test']}
    encoding_variants = []
    for job in sorted(payloads):
        config, split, offset = job
        code = next(x for x in config.split('-') if x != 'en')
        for item in payloads[job]['rows']:
            translation = item['row']['translation']
            parent = f'opus:{config}:{split}:{item["row_idx"]}'
            for language in [code, 'en']:
                text = norm(translation[language])
                label = code_map[language]
                original_text = text
                if split == 'train' and label == 'isl':
                    # 部分旧字幕使用错误字形，训练端修复；外部测试保持原文。
                    repaired = text.translate(str.maketrans('đĐūŪķĶũŨõ', 'ðÐþÞóÓýÝð'))
                    stats['icelandic_training_encoding_repairs'] += repaired != text
                    text = repaired
                if not usable(text):
                    stats[f'opus_{split}_too_short_or_long'] += 1
                    continue
                if split == 'train' and not consistent(text, label):
                    stats['opus_train_teacher_disagreement'] += 1
                    continue
                candidates[split][label].append(row(text, label, 'opus-100', parent))
                if split == 'train' and label == 'isl' and original_text != text:
                    encoding_variants.append({**row(original_text, label, 'legacy_encoding_variant', parent),
                                              'synthetic': True})

    # WiLI 补充长文本及 OPUS 未覆盖的类别；只筛训练池，不拿教师筛选测试答案。
    wmeta = json.loads((ROOT / 'data/wili_metadata.json').read_text())
    names = wmeta['cardData']['dataset_info']['features'][1]['dtype']['class_label']['names']
    wiki = defaultdict(list)
    for item in pq.read_table(ROOT / 'data/train.parquet').to_pylist():
        label = names[str(item['label'])]
        text = norm(item['sentence'])
        if label not in labels or not usable(text):
            continue
        if not consistent(text, label):
            stats['wiki_teacher_disagreement'] += 1
            continue
        wiki[label].append(row(text, label, 'wili-train', 'wiki:' + fingerprint(text)))

    # 全局先分配真实测试，再验证，再训练；同时检查完整文本和 96-token 输入。
    used_text, used_tokens, heldout_parents = set(), set(), set()
    def token_key(text):
        ids = tokenizer.encode(text, truncation=True, max_length=96)
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    def claim(record, heldout=False):
        key = token_key(record['text'])
        if heldout and (record['hash'] in baseline_seen or key in baseline_token_inputs):
            stats['previous_experiment_seen_excluded_from_holdout'] += 1
            return False
        if record['hash'] in used_text or key in used_tokens:
            stats['duplicate_removed'] += 1
            return False
        used_text.add(record['hash']); used_tokens.add(key)
        if heldout:
            heldout_parents.add(record['parent'])
        return True
    result = {s: [] for s in ['train', 'validation', 'test']}
    for split, cap in [('test', 100), ('validation', 50), ('train', 400)]:
        for label in labels:
            items = candidates[split][label]
            rng.shuffle(items)
            taken = 0
            for record in items:
                if record['parent'] in heldout_parents:
                    continue
                if claim(record, heldout=split != 'train'):
                    result[split].append(record)
                    taken += 1
                if taken >= cap:
                    break
    # 对 OPUS 缺失类别，保留独立 Wiki 验证；外部测试明确不覆盖这些类别。
    validation_labels = {r['label'] for r in result['validation']}
    for label in labels:
        items = wiki[label]
        rng.shuffle(items)
        if label not in validation_labels:
            for record in items:
                if claim(record, heldout=True):
                    result['validation'].append(record)
                    if sum(r['label'] == label for r in result['validation']) >= 30:
                        break
        added = 0
        for record in items:
            if record['parent'] in heldout_parents:
                continue
            if claim(record):
                result['train'].append(record)
                added += 1
            if added >= 150:
                break
    real_training = list(result['train'])
    real_parents = {r['parent'] for r in real_training}
    for record in encoding_variants:
        if record['parent'] in real_parents and claim(record):
            result['train'].append(record)
            stats['legacy_encoding_training_variants'] += 1
    # 仅训练集：从通过教师检查的父文本生成短片段或轻扰动，不跨集合制造派生泄漏。
    generated = Counter()
    for record in real_training:
        label = record['label']
        if generated[label] >= 150:
            continue
        text = record['text']
        if len(text) > 60:
            n = rng.choice([24, 32, 40, 56])
            start = rng.randint(0, max(0, len(text) - n))
            variant = text[start:start+n].strip()
            kind = 'random_character_crop'
        else:
            variant = text.rstrip('.!?。！？') + rng.choice(['!', '…', ' 2026', ' :)'])
            kind = 'punctuation_or_number'
        if not usable(variant) or not consistent(variant, label):
            stats['synthetic_rejected'] += 1
            continue
        new = {**record, 'text': variant, 'hash': fingerprint(variant),
               'source': kind, 'synthetic': True, 'parent': record['parent']}
        if claim(new):
            result['train'].append(new)
            generated[label] += 1
    output = ROOT / 'data/expanded'
    output.mkdir(exist_ok=True)
    for split, records in result.items():
        rng.shuffle(records)
        with (output / f'{split}.jsonl').open('w') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    train_labels = {r['label'] for r in result['train']}
    if set(labels) != train_labels:
        raise RuntimeError(f'训练缺失类别 {set(labels)-train_labels}，不能声称覆盖 109 类')
    summary = {'counts': {s: dict(Counter(r['label'] for r in rows)) for s, rows in result.items()},
               'samples': {s: len(v) for s, v in result.items()},
               'synthetic_training': sum(r['synthetic'] for r in result['train']),
               'training_languages': len(train_labels),
               'test_languages': len({r['label'] for r in result['test']}),
               'missing_external_test': sorted(set(labels)-{r['label'] for r in result['test']}),
               'teacher_filter': 'FastText lid.176.ftz，仅删除分数>=0.75的非近缘不一致训练条目；保留不确定项，不重标，不筛测试',
               'teacher_family_exceptions': [['bs','hr','sr','sh'], ['su','jv','id','ms'], ['ht','fr']],
               'source_license': {'wili': 'ODbL', 'opus-100': '数据集卡 unknown，底层语料条款各异；实验数据不重新分发'},
               'dataset_metadata_sha': meta['sha'], 'sampling_seed': 20261021,
               'file_revisions': sorted(revisions),
               'snapshot_note': '从固定提交读取 Parquet 字节范围，检查 HTTP x-repo-commit；保留范围缓存与逐条指纹',
               'statistics': dict(stats)}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
