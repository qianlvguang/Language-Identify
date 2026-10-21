"""从固定提交的 Parquet 文件按字节范围读取少量行组，避免预览 API。"""
import hashlib
import io
import json
from pathlib import Path
import random
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

class RemoteFile(io.RawIOBase):
    def __init__(self, url, cache, revision):
        super().__init__()
        self.url, self.cache, self.revision = url, cache, revision
        self.cache.mkdir(parents=True, exist_ok=True)
        self.position = 0
        _, headers = self.fetch(0, 1)
        self.size = int(re.findall(r'content-range: bytes \d+-\d+/(\d+)', headers, re.I)[-1])

    def fetch(self, offset, n):
        path = self.cache / f'{offset}_{n}.bin'
        head = path.with_suffix('.headers')
        if not path.exists():
            temp = path.with_suffix('.partial')
            subprocess.run(['curl', '-fsSL', '--retry', '2', '--connect-timeout', '15',
                            '--max-time', '60', '-H', f'Range: bytes={offset}-{offset+n-1}',
                            '-D', str(head), '-o', str(temp), self.url], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if temp.stat().st_size != n:
                raise RuntimeError(f'服务器未按范围返回：期望 {n}，实际 {temp.stat().st_size}')
            temp.rename(path)
        headers = head.read_text()
        assert self.revision in headers, '文件响应未包含所请求的固定提交'
        return path.read_bytes(), headers

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.position
        n = min(n, self.size - self.position)
        if n <= 0:
            return b''
        data, _ = self.fetch(self.position, n)
        self.position += n
        return data

def page(job):
    import pyarrow.parquet as pq
    config, split, _ = job
    revision = json.loads((ROOT / 'data/opus_metadata.json').read_text())['sha']
    output = ROOT / 'data/opus_file_samples' / config / f'{split}.json'
    if output.exists():
        return job, json.loads(output.read_text())
    url = f'https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/{revision}/{config}/{split}-00000-of-00001.parquet'
    remote = RemoteFile(url, ROOT / 'data/opus_ranges' / config / split, revision)
    file = pq.ParquetFile(remote)
    rng = random.Random(int(hashlib.sha256(f'{config}:{split}:42'.encode()).hexdigest()[:8], 16))
    groups = rng.sample(range(file.num_row_groups), min(2 if split == 'train' else 1, file.num_row_groups))
    rows = []
    for group in groups:
        start = sum(file.metadata.row_group(i).num_rows for i in range(group))
        data = file.read_row_group(group).to_pylist()
        indices = list(range(len(data)))
        rng.shuffle(indices)
        take = 200 if split == 'train' else 300
        rows.extend({'row_idx': start+i, 'row': data[i], 'truncated_cells': []} for i in indices[:take])
    payload = {'rows': rows, 'revision': revision, 'row_groups': groups, 'url': url}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False))
    return job, payload

if __name__ == '__main__':
    job, payload = page(('en-is', 'train', 0))
    print(json.dumps({'job': job, 'rows': len(payload['rows']), 'sample': payload['rows'][0]}, ensure_ascii=False))
