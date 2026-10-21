"""检查训练/验证/测试哈希隔离，以及模型截断后输入的精确重合。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import unicodedata

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'data/hf_cache'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/local_lora_109')
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    config = json.loads((args.run / 'config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/xlm_roberta_base', local_files_only=True)
    texts = {}
    for split in ['test', 'train']:
        for row in pq.read_table(ROOT / 'data' / f'{split}.parquet', columns=['sentence']).to_pylist():
            text = unicodedata.normalize('NFC', row['sentence']).strip()
            h = hashlib.sha256(' '.join(text.split()).casefold().encode()).hexdigest()
            texts.setdefault(h, text)
    hashes, token_hashes, token_rows = {}, {}, {}
    sizes = {}
    for split in ['train', 'validation', 'test']:
        rows = [json.loads(s) for s in (args.run / f'{split}_manifest.jsonl').read_text().splitlines()]
        hashes[split] = {r['hash'] for r in rows}
        token_hashes[split] = set()
        token_rows[split] = []
        for start in range(0, len(rows), 512):
            batch = rows[start:start+512]
            encoded = tokenizer([texts[r['hash']] for r in batch], truncation=True,
                                max_length=config['max_length'])['input_ids']
            for row, ids in zip(batch, encoded):
                token_digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                token_hashes[split].add(token_digest)
                token_rows[split].append((row['hash'], token_digest))
        sizes[split] = {'rows': len(rows), 'unique_hashes': len(hashes[split]),
                        'unique_truncated_inputs': len(token_hashes[split])}
    result = {'sizes': sizes, 'full_text_overlap': {}, 'truncated_input_overlap': {}}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        result['full_text_overlap'][a + '/' + b] = len(hashes[a] & hashes[b])
        result['truncated_input_overlap'][a + '/' + b] = len(token_hashes[a] & token_hashes[b])
    result['exclude_for_clean_evaluation'] = {
        'validation': [h for h, digest in token_rows['validation'] if digest in token_hashes['train']],
        'test': [h for h, digest in token_rows['test']
                 if digest in token_hashes['train'] or digest in token_hashes['validation']]}
    (args.run / 'split_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
