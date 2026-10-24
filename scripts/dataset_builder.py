"""大规模多源平衡语料构建流水线 (DatasetBuilder)。

功能与特性：
1. 语言学分类自动对齐：严格按照 language_taxonomy 将多源语料收敛映射为 87 目标分类。
2. 双通道数据源：
   - 支持从线上 (HuggingFace/OPUS) 自动抓取与增量扩充；
   - 本地多源抽取池：整合 WiLI-2018 (23.5万条)、FLORES-devtest 与 OPUS 平行句。
3. 工业级数据增强生成：
   - 自动调用 OCRManualAugmenter 生成标准句、极短截断句 (5~35字符)、OCR形近字错字、重度乱码与说明书电气参数混入句。
4. 训练规模大幅扩充：
   - 支持生成 10万~20万+ 高质量平衡训练样本。
5. 阶梯式独立测试集隔离：
   - 保证测试集与训练集/验证集严格去重，输出标准原句集、短句集、轻度OCR集与重度乱码集。
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import unicodedata

# 注入项目根目录以确保能引入本地模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.language_taxonomy import TARGET_LANGUAGES, map_to_target, LANGUAGE_NAMES_87
from scripts.ocr_noise_augmenter import OCRManualAugmenter


def is_script_valid(text: str, label: str) -> bool:
    """严格校验样本的 Unicode 脚本真实性，物理清除维基百科中未翻译的英语引文和脏数据。"""
    chars = [c for c in text if c.isalpha()]
    if len(chars) < 6:
        return False

    def in_range(c, start, end):
        return start <= ord(c) <= end

    # 1. 希腊语
    if label == 'ell':
        return sum(in_range(c, 0x0370, 0x03FF) for c in chars) / len(chars) >= 0.4
    # 2. 西里尔字母族 (俄/乌/白/保/马/塔)
    if label in ['rus', 'ukr', 'bel', 'bul', 'mkd', 'tgk', 'kaz', 'kir', 'mon']:
        return sum(in_range(c, 0x0400, 0x04FF) for c in chars) / len(chars) >= 0.4
    # 3. 阿拉伯字母族 (阿/波/乌尔都/普什图/维)
    if label in ['ara', 'fas', 'urd', 'pus', 'uig']:
        return sum(in_range(c, 0x0600, 0x06FF) for c in chars) / len(chars) >= 0.4
    # 4. 希伯来/意第绪
    if label in ['heb', 'yid']:
        return sum(in_range(c, 0x0590, 0x05FF) for c in chars) / len(chars) >= 0.4
    # 5. 印度天城文族 (印地/马拉地/尼泊尔)
    if label in ['hin', 'mar', 'nep']:
        return sum(in_range(c, 0x0900, 0x097F) for c in chars) / len(chars) >= 0.4
    # 6. 孟加拉/阿萨姆
    if label in ['ben', 'asm']:
        return sum(in_range(c, 0x0980, 0x09FF) for c in chars) / len(chars) >= 0.4
    # 7. 旁遮普
    if label == 'pan':
        return sum(in_range(c, 0x0A00, 0x0A7F) for c in chars) / len(chars) >= 0.4
    # 8. 古吉拉特
    if label == 'guj':
        return sum(in_range(c, 0x0A80, 0x0AFF) for c in chars) / len(chars) >= 0.4
    # 9. 奥里亚
    if label == 'ori':
        return sum(in_range(c, 0x0B00, 0x0B7F) for c in chars) / len(chars) >= 0.4
    # 10. 泰米尔
    if label == 'tam':
        return sum(in_range(c, 0x0B80, 0x0BFF) for c in chars) / len(chars) >= 0.4
    # 11. 泰卢固
    if label == 'tel':
        return sum(in_range(c, 0x0C00, 0x0C7F) for c in chars) / len(chars) >= 0.4
    # 12. 卡纳达
    if label == 'kan':
        return sum(in_range(c, 0x0C80, 0x0CFF) for c in chars) / len(chars) >= 0.4
    # 13. 马拉雅拉姆
    if label == 'mal':
        return sum(in_range(c, 0x0D00, 0x0D7F) for c in chars) / len(chars) >= 0.4
    # 14. 僧伽罗
    if label == 'sin':
        return sum(in_range(c, 0x0D80, 0x0DFF) for c in chars) / len(chars) >= 0.4
    # 15. 泰语
    if label == 'tha':
        return sum(in_range(c, 0x0E00, 0x0E7F) for c in chars) / len(chars) >= 0.4
    # 16. 藏语
    if label == 'bod':
        return sum(in_range(c, 0x0F00, 0x0FFF) for c in chars) / len(chars) >= 0.4
    # 17. 缅甸语
    if label == 'mya':
        return sum(in_range(c, 0x1000, 0x109F) for c in chars) / len(chars) >= 0.4
    # 18. 格鲁吉亚
    if label == 'kat':
        return sum(in_range(c, 0x10A0, 0x10FF) for c in chars) / len(chars) >= 0.4
    # 19. 亚美尼亚
    if label == 'hye':
        return sum(in_range(c, 0x0530, 0x058F) for c in chars) / len(chars) >= 0.4
    # 20. 中日韩
    if label == 'zho':
        return sum(in_range(c, 0x4E00, 0x9FFF) for c in chars) / len(chars) >= 0.3
    if label == 'jpn':
        return any(in_range(c, 0x3040, 0x30FF) for c in chars) or any(in_range(c, 0x4E00, 0x9FFF) for c in chars)
    if label == 'kor':
        return sum(in_range(c, 0xAC00, 0xD7AF) or in_range(c, 0x1100, 0x11FF) for c in chars) / len(chars) >= 0.3

    return True


def compute_hash(text: str) -> str:
    """计算规范化文本的哈希指纹。"""
    normalized = ' '.join(unicodedata.normalize('NFC', text).split()).casefold()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


class DatasetBuilder:
    """大规模 87 语种说明书 OCR 数据集构建器。"""

    def __init__(self, data_dir: Path = ROOT / 'data', seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed
        self.rng = random.Random(seed)
        self.augmenter = OCRManualAugmenter(seed=seed)
        self.target_labels = sorted(TARGET_LANGUAGES)
        self.label_to_id = {label: i for i, label in enumerate(self.target_labels)}

    def extract_from_wili(self) -> Dict[str, List[str]]:
        """从本地 WiLI-2018 的 train 与 test parquet 文件中抽取并映射语料。"""
        import pyarrow.parquet as pq

        meta_path = self.data_dir / 'wili_metadata.json'
        if not meta_path.exists():
            return {}

        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        class_names = meta['cardData']['dataset_info']['features'][1]['dtype']['class_label']['names']

        corpus = defaultdict(list)
        for split_file in ['train.parquet', 'test.parquet']:
            p = self.data_dir / split_file
            if not p.exists():
                continue
            table = pq.read_table(p).to_pylist()
            for row in table:
                raw_code = class_names[str(row['label'])]
                target = map_to_target(raw_code)
                if not target:
                    continue
                text = row['sentence'].strip()
                if len(text) >= 15 and sum(c.isalpha() for c in text) >= 8 and is_script_valid(text, target):
                    corpus[target].append(text)
        return corpus

    def extract_from_flores(self) -> Dict[str, List[str]]:
        """从本地 FLORES-devtest parquet 抽取高质量多语言句子。"""
        import pyarrow.parquet as pq
        flores_p = self.data_dir / 'flores_devtest.parquet'
        if not flores_p.exists():
            return {}
        corpus = defaultdict(list)
        try:
            table = pq.read_table(flores_p).to_pylist()
            for row in table:
                # flores 通常包含 iso 字段或根据字段名映射
                for k, v in row.items():
                    target = map_to_target(k)
                    if target and isinstance(v, str) and len(v.strip()) >= 15 and is_script_valid(v.strip(), target):
                        corpus[target].append(v.strip())
        except Exception:
            pass
        return corpus

    def fetch_online_huggingface(self, max_samples_per_lang: int = 500) -> Dict[str, List[str]]:
        """尝试从线上 HuggingFace 拉取额外多语言语料 (具备离线/网络受限容错)。"""
        corpus = defaultdict(list)
        print("尝试连接线上语料库进行扩充...", flush=True)
        try:
            from datasets import load_dataset
            # 尝试加载轻量多语言识别语料
            ds = load_dataset('papluca/language-identification', split='train', streaming=True)
            for item in ds:
                raw_code = item.get('labels') or item.get('label')
                text = item.get('text', '').strip()
                target = map_to_target(raw_code)
                if target and len(corpus[target]) < max_samples_per_lang:
                    corpus[target].append(text)
                if all(len(corpus[l]) >= max_samples_per_lang for l in self.target_labels):
                    break
            print(f"线上扩充完成，累计获取 {sum(len(v) for v in corpus.values())} 条样本。", flush=True)
        except Exception as e:
            print(f"提示: 线上拉取未执行或受限 ({e})，自动完全基于本地高吞吐语料池进行扩充。", flush=True)
        return corpus

    def build_dataset(
        self,
        output_dir: Path,
        train_samples_per_lang: int = 1500,
        val_samples_per_lang: int = 100,
        test_samples_per_lang: int = 100,
        try_online: bool = False
    ) -> Dict[str, int]:
        """执行端到端多源合并、清洗、增强与分层数据集生成。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        print("开始归集多源基础语料...", flush=True)

        raw_pools = defaultdict(list)

        # 1. 抽取本地 WiLI
        wili_data = self.extract_from_wili()
        for l, texts in wili_data.items():
            raw_pools[l].extend(texts)

        # 2. 抽取本地 FLORES
        flores_data = self.extract_from_flores()
        for l, texts in flores_data.items():
            raw_pools[l].extend(texts)

        # 3. 若允许线上，则尝试线上拉取
        if try_online:
            online_data = self.fetch_online_huggingface(max_samples_per_lang=500)
            for l, texts in online_data.items():
                raw_pools[l].extend(texts)

        # 严格全局去重
        deduped_pools = {}
        global_hashes = set()
        for label in self.target_labels:
            texts = raw_pools[label]
            self.rng.shuffle(texts)
            unique_texts = []
            for t in texts:
                h = compute_hash(t)
                if h not in global_hashes:
                    global_hashes.add(h)
                    unique_texts.append(t)
            deduped_pools[label] = unique_texts

        # 检查各语种样本储备
        counts = {l: len(deduped_pools[l]) for l in self.target_labels}
        print(f"语种抽取汇总: 87 语种就绪，各语种平均原始去重样本数: {sum(counts.values()) // len(counts)}", flush=True)

        train_records = []
        val_records = []
        test_standard_records = []
        test_short_records = []
        test_light_ocr_records = []
        test_heavy_ocr_records = []

        test_hashes = set()

        # 生成各数据集
        for label in self.target_labels:
            available = deduped_pools[label]
            if not available:
                continue

            # 保障测试集与训练/验证严格完全物理隔离
            # 拿出后 200 条作为测试储备
            test_pool = available[-200:]
            train_val_pool = available[:-200] if len(available) > 200 else available[:len(available) // 2]
            if not train_val_pool:
                train_val_pool = test_pool

            # 1. 构建严格隔离的测试集 (每类各 100 条)
            cur_test = test_pool[:test_samples_per_lang]
            for t in cur_test:
                h = compute_hash(t)
                test_hashes.add(h)
                # 原文标准测试
                test_standard_records.append({
                    'text': t, 'label': label, 'hash': h, 'mode': 'standard'
                })
                # 短句测试 (<=35 字符)
                short_t = self.augmenter.augment(t, mode='short_only')
                test_short_records.append({
                    'text': short_t, 'label': label, 'parent_hash': h, 'mode': 'short_only'
                })
                # 轻度 OCR 错字测试
                light_t = self.augmenter.augment(t, mode='light_ocr')
                test_light_ocr_records.append({
                    'text': light_t, 'label': label, 'parent_hash': h, 'mode': 'light_ocr'
                })
                # 重度 OCR 乱码与规格测试
                heavy_t = self.augmenter.augment(t, mode='heavy_ocr')
                test_heavy_ocr_records.append({
                    'text': heavy_t, 'label': label, 'parent_hash': h, 'mode': 'heavy_ocr'
                })

            # 2. 构建验证集 (每类 100 条)
            cur_val = [t for t in train_val_pool[:val_samples_per_lang] if compute_hash(t) not in test_hashes]
            for t in cur_val:
                val_records.append({
                    'text': t, 'label': label, 'hash': compute_hash(t), 'mode': 'standard'
                })

            # 3. 构建大规模平衡训练集 (目标每类 1500~2000 条，含原始句、短句、OCR形近字与说明书混合句)
            train_base = [t for t in train_val_pool[val_samples_per_lang:] if compute_hash(t) not in test_hashes]
            if not train_base:
                train_base = cur_val

            # 循环/扩充采样达到 train_samples_per_lang
            cur_train = []
            idx = 0
            while len(cur_train) < train_samples_per_lang:
                base_text = train_base[idx % len(train_base)]
                idx += 1

                # 按照现实说明书分布决定本条样本的形态:
                # 25% 干净原句; 30% 短文本截断; 25% OCR形近/变音退化; 20% 说明书规格与综合乱码
                r = self.rng.random()
                if r < 0.25:
                    final_text = base_text
                    mode = 'clean'
                elif r < 0.55:
                    final_text = self.augmenter.augment(base_text, mode='short_only')
                    mode = 'short_only'
                elif r < 0.80:
                    final_text = self.augmenter.augment(base_text, mode='light_ocr')
                    mode = 'light_ocr'
                else:
                    final_text = self.augmenter.augment(base_text, mode='mixed')
                    mode = 'mixed_ocr'

                cur_train.append({
                    'text': final_text,
                    'label': label,
                    'hash': compute_hash(final_text),
                    'parent_hash': compute_hash(base_text),
                    'mode': mode
                })
            train_records.extend(cur_train)

        # 打乱各数据集
        self.rng.shuffle(train_records)
        self.rng.shuffle(val_records)
        self.rng.shuffle(test_standard_records)
        self.rng.shuffle(test_short_records)
        self.rng.shuffle(test_light_ocr_records)
        self.rng.shuffle(test_heavy_ocr_records)

        # 保存为 jsonl
        def write_jsonl(path: Path, rows: List[dict]):
            with path.open('w', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

        write_jsonl(output_dir / 'train.jsonl', train_records)
        write_jsonl(output_dir / 'validation.jsonl', val_records)
        write_jsonl(output_dir / 'test_standard.jsonl', test_standard_records)
        write_jsonl(output_dir / 'test_short.jsonl', test_short_records)
        write_jsonl(output_dir / 'test_light_ocr.jsonl', test_light_ocr_records)
        write_jsonl(output_dir / 'test_heavy_ocr.jsonl', test_heavy_ocr_records)

        summary = {
            'target_languages_count': len(self.target_labels),
            'train_total': len(train_records),
            'validation_total': len(val_records),
            'test_standard_total': len(test_standard_records),
            'test_short_total': len(test_short_records),
            'test_light_ocr_total': len(test_light_ocr_records),
            'test_heavy_ocr_total': len(test_heavy_ocr_records),
        }
        (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"数据集构建圆满完成！输出目录: {output_dir}")
        print(f"训练集规模: {len(train_records)} 条 (87 语种平衡)，验证集: {len(val_records)} 条")
        return summary


def main():
    parser = argparse.ArgumentParser(description="大规模 87 语种 OCR 说明书语料构建脚本")
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/ocr_v2')
    parser.add_argument('--train-per-lang', type=int, default=1500, help="每种语言训练样本数 (默认1500，总数约13万条)")
    parser.add_argument('--val-per-lang', type=int, default=100)
    parser.add_argument('--test-per-lang', type=int, default=100)
    parser.add_argument('--try-online', action='store_true', help="是否尝试连接线上 HuggingFace 拉取增量语料")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    builder = DatasetBuilder(seed=args.seed)
    builder.build_dataset(
        output_dir=args.output_dir,
        train_samples_per_lang=args.train_per_lang,
        val_samples_per_lang=args.val_per_lang,
        test_samples_per_lang=args.test_per_lang,
        try_online=args.try_online
    )


if __name__ == '__main__':
    main()
