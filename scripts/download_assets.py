"""下载固定版本的公开模型与小型语言识别数据，文件仅保存在项目内。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]

def metadata(repo, kind, revision, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/api/{kind}/{repo}/revision/{revision}'
        subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '60',
                        '-o', str(path), url], check=True)
    result = json.loads(path.read_text())
    assert result['sha'] == revision
    return result

def download(item):
    url, path = item
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.partial')
        subprocess.run(['curl', '-fL', '--retry', '3', '--connect-timeout', '30',
                        '--max-time', '1200', '-o', str(partial), url], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        partial.rename(path)
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    print(json.dumps({'file': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                      'sha256': digest}), flush=True)
    return {'file': str(path.relative_to(ROOT)), 'sha256': digest, 'url': url}

if __name__ == '__main__':
    model = metadata('FacebookAI/xlm-roberta-base', 'models',
                     'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089', ROOT / 'models/base_metadata.json')
    data = metadata('MartinThoma/wili_2018', 'datasets',
                    '12c3a519f58673e4051a5d9db51c1fcd94a94c75', ROOT / 'data/wili_metadata.json')
    metadata('Helsinki-NLP/opus-100', 'datasets',
             '805090dc28bf78897da9641cdf08b61287580df9', ROOT / 'data/opus_metadata.json')
    items = [(f'https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/{model["sha"]}/{f}',
              ROOT / 'models/xlm_roberta_base' / f)
             for f in ['config.json', 'model.safetensors', 'tokenizer.json',
                       'tokenizer_config.json', 'sentencepiece.bpe.model']]
    items += [(f'https://huggingface.co/datasets/MartinThoma/wili_2018/resolve/{data["sha"]}/'
               + quote(f'WiLI-2018 dataset/{split}-00000-of-00001.parquet'),
               ROOT / 'data' / f'{split}.parquet') for split in ['train', 'test']]
    items.append(('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz',
                  ROOT / 'models/lid.176.ftz'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(download, items))
    (ROOT / 'data/download_manifest.json').write_text(json.dumps(result, indent=2))
