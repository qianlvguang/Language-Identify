"""OCR 说明书退化仿真与单条短文本数据增强器。

针对真实业务场景中说明书单条 OCR 识别的 5 大典型缺陷进行高拟真仿真：
1. 极短文本/片段截断 (5~35 字符)
2. OCR 形近字符替换与常见错别字 (如 rn<->m, cl<->d, 0<->O, l<->1)
3. 变音符号剥离、退化与 UTF-8 乱码模拟 (ü->u, é->e, Ã¼ 等)
4. 排版与空白错乱 (漏空格、多余散字空格、换行折词连字符残留)
5. 说明书专有电气参数与通用规格混入 (220V 50Hz, 3000mAh, IPX4, USB, Model 等)
"""
import random
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# 跨语言形近字符置换表
HOMOGLYPH_REPLACEMENTS: Dict[str, List[str]] = {
    # 拉丁字符形近与大小写混淆
    'rn': ['m'],
    'm': ['rn'],
    'cl': ['d'],
    'd': ['cl'],
    'vv': ['w'],
    'w': ['vv'],
    'l': ['1', 'I', '|'],
    '1': ['l', 'I', '|'],
    'I': ['l', '1', '|'],
    '0': ['O', 'o'],
    'O': ['0', 'o'],
    'o': ['0'],
    'c': ['e', '('],
    'e': ['c'],
    'B': ['8', '|3'],
    '8': ['B'],
    'S': ['5', '$'],
    '5': ['S'],
    'Z': ['2'],
    '2': ['Z'],
    'g': ['q', '9'],
    'q': ['g', '9'],
    'u': ['v'],
    'v': ['u'],
    # 西里尔与拉丁混淆 (常见于东欧/俄语/乌克兰语 OCR)
    'с': ['c'], 'а': ['a'], 'о': ['o', '0'], 'е': ['e'],
    'р': ['p'], 'х': ['x'], 'у': ['y'], 'в': ['b', 'B'],
}

# 变音符号剥离字典 (模拟低精度 OCR 将变音字符识别为基础英文字符)
DIACRITICS_STRIP_MAP: Dict[str, str] = {
    'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a', 'ā': 'a', 'æ': 'ae',
    'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e', 'ē': 'e', 'ě': 'e',
    'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i', 'ī': 'i',
    'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o', 'ø': 'o', 'ō': 'o',
    'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u', 'ū': 'u',
    'ý': 'y', 'ÿ': 'y',
    'ç': 'c', 'č': 'c', 'ć': 'c',
    'ñ': 'n', 'ń': 'n', 'ň': 'n',
    'š': 's', 'ś': 's', 'ş': 's',
    'ž': 'z', 'ź': 'z', 'ż': 'z',
    'ł': 'l', 'đ': 'd', 'ð': 'd', 'þ': 'th', 'ß': 'ss',
    # 大写变音
    'Á': 'A', 'À': 'A', 'Â': 'A', 'Ä': 'A', 'Å': 'A',
    'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
    'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
    'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Ö': 'O', 'Ø': 'O',
    'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
    'Ç': 'C', 'Ñ': 'N', 'Š': 'S', 'Ž': 'Z'
}

# 常见 UTF-8 错乱乱码映射 (Windows-1252 / ISO-8859 误解码)
MOJIOBAKE_MAP: Dict[str, str] = {
    'é': 'Ã©', 'è': 'Ã¨', 'ê': 'Ãª', 'ë': 'Ã«',
    'à': 'Ã\xa0', 'â': 'Ã¢', 'ä': 'Ã¤', 'å': 'Ã¥',
    'ü': 'Ã¼', 'ö': 'Ã¶', 'ä': 'Ã¤', 'ß': 'Ã\x9f',
    'ñ': 'Ã±', 'ç': 'Ã§', 'ø': 'Ã¸', 'æ': 'Ã¦',
    '’': 'â€™', '“': 'â€œ', '”': 'â€\x9d', '—': 'â€”'
}

# 常见扫描杂点符号
SCAN_ARTIFACTS: List[str] = [
    '_', '^', '~', '|', '/', '\\', '?', '*', '■', '●', '°', '§', '¢',
    '©', '®', '™', '¬', '•', '·', '—', '-', ':', ';'
]

# 说明书常见电气、规格与中立参数片段
MANUAL_SPEC_SNIPPETS: List[str] = [
    '220V 50Hz', '100-240V~ 50/60Hz', 'DC 5V/2A', '3000mAh', '12V 1.5A',
    'Max 60°C', 'IPX4 Waterproof', 'Model: XY-2024', 'Type-C USB', 'LED Indicator',
    'Bluetooth 5.3', 'WiFi 2.4GHz', 'RoHS CE FCC', 'Step 1:', 'Fig. 2',
    'No. 0819', 'Power ON/OFF', 'Reset Button', 'Made in PRC', 'SN: A92-X10'
]


class OCRManualAugmenter:
    """工业级 OCR 说明书退化仿真器。"""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def slice_short_text(self, text: str, min_len: int = 8, max_len: int = 35) -> str:
        """从较长文本中随机切片出 8~35 字符的短句或词组，模拟说明书单条截断。"""
        text = text.strip()
        if len(text) <= min_len:
            return text
        target_len = self.rng.randint(min_len, min(max_len, len(text)))
        max_start = max(0, len(text) - target_len)
        start = self.rng.randint(0, max_start)
        sliced = text[start:start + target_len].strip()
        # 避免切出纯标点或纯空白
        if sum(c.isalpha() for c in sliced) < 3:
            return text[:target_len].strip()
        return sliced

    def apply_homoglyphs(self, text: str, prob: float = 0.25) -> str:
        """替换形近字符与 OCR 易错拼写。"""
        # 先处理多字符置换 (rn, cl, vv)
        for multi, targets in [('rn', ['m']), ('cl', ['d']), ('vv', ['w'])]:
            if multi in text and self.rng.random() < prob:
                text = text.replace(multi, self.rng.choice(targets), 1)

        result = []
        for ch in text:
            if ch in HOMOGLYPH_REPLACEMENTS and self.rng.random() < prob:
                result.append(self.rng.choice(HOMOGLYPH_REPLACEMENTS[ch]))
            else:
                result.append(ch)
        return ''.join(result)

    def apply_diacritics_corruption(self, text: str, prob: float = 0.35) -> str:
        """模拟说明书 OCR 中变音符号丢失或乱码。"""
        result = []
        for ch in text:
            if self.rng.random() < prob:
                if ch in MOJIOBAKE_MAP and self.rng.random() < 0.2:
                    result.append(MOJIOBAKE_MAP[ch])
                elif ch in DIACRITICS_STRIP_MAP:
                    result.append(DIACRITICS_STRIP_MAP[ch])
                else:
                    result.append(ch)
            else:
                result.append(ch)
        return ''.join(result)

    def apply_whitespace_flaws(self, text: str) -> str:
        """模拟排版空格丢失 (连词) 或空格过多 (散字) 或换行连字符。"""
        mode = self.rng.choice(['missing_space', 'excess_space', 'hyphen_break', 'case_mess', 'none'])
        if mode == 'missing_space' and ' ' in text:
            # 随机删去一个空格造成连词 (如 Caution:Hot)
            words = text.split(' ')
            if len(words) >= 2:
                idx = self.rng.randint(0, len(words) - 2)
                words[idx] = words[idx] + words[idx + 1]
                words.pop(idx + 1)
                return ' '.join(words)
        elif mode == 'excess_space' and len(text) > 5 and ' ' not in text:
            # 在短词内部插入空格 (如 W a r n i n g)
            return ' '.join(list(text))
        elif mode == 'hyphen_break' and len(text) > 10 and ' ' in text:
            # 模拟折行连字符残留 (如 opera- tion)
            words = text.split()
            idx = self.rng.randint(0, len(words) - 1)
            target = words[idx]
            if len(target) >= 6:
                cut = self.rng.randint(2, len(target) - 2)
                words[idx] = target[:cut] + '- ' + target[cut:]
                return ' '.join(words)
        elif mode == 'case_mess':
            # 模拟全大写或首字母畸变
            if self.rng.random() < 0.5:
                return text.upper()
            else:
                return text.title()
        return text

    def apply_scanned_noise(self, text: str, noise_count: int = 1) -> str:
        """随机注入扫描杂点符号或乱码。"""
        if not text:
            return text
        chars = list(text)
        for _ in range(noise_count):
            artifact = self.rng.choice(SCAN_ARTIFACTS)
            pos = self.rng.randint(0, len(chars))
            chars.insert(pos, artifact)
        return ''.join(chars)

    def inject_manual_specs(self, text: str) -> str:
        """混入说明书常见规格、电气参数、技术英文词。"""
        spec = self.rng.choice(MANUAL_SPEC_SNIPPETS)
        mode = self.rng.choice(['prefix', 'suffix', 'bracket'])
        if mode == 'prefix':
            return f"[{spec}] {text}"
        elif mode == 'suffix':
            return f"{text} ({spec})"
        else:
            return f"{text} - {spec}"

    def augment(self, text: str, mode: str = 'mixed') -> str:
        """主入口：对单条文本进行 OCR 说明书退化仿真。
        
        mode 选项:
        - 'clean': 保持原样 (仅规范化)
        - 'short_only': 仅进行短文本截断 (5~35 字符)
        - 'light_ocr': 轻度 OCR 扰动 (形近字 + 空白轻微扰动)
        - 'heavy_ocr': 重度 OCR 乱码 (变音丢失 + 扫描杂点 + 错乱)
        - 'spec_mixed': 混入说明书规格参数与短文本截断
        - 'mixed': 组合策略 (按照现实分布综合退化)
        """
        text = unicodedata.normalize('NFC', text).strip()
        if not text:
            return text

        if mode == 'clean':
            return text
        elif mode == 'short_only':
            return self.slice_short_text(text)
        elif mode == 'light_ocr':
            t = self.apply_homoglyphs(text, prob=0.2)
            t = self.apply_whitespace_flaws(t)
            return t
        elif mode == 'heavy_ocr':
            t = self.apply_diacritics_corruption(text, prob=0.4)
            t = self.apply_homoglyphs(t, prob=0.3)
            t = self.apply_scanned_noise(t, noise_count=self.rng.randint(1, 3))
            t = self.apply_whitespace_flaws(t)
            return t
        elif mode == 'spec_mixed':
            t = self.slice_short_text(text, min_len=6, max_len=30)
            t = self.inject_manual_specs(t)
            return t

        # 'mixed' 默认综合退化策略
        # 1. 50% 概率进行短文本截断
        if self.rng.random() < 0.5:
            text = self.slice_short_text(text)

        # 2. 40% 概率混入说明书电气/规格参数
        if self.rng.random() < 0.35:
            text = self.inject_manual_specs(text)

        # 3. 50% 概率进行变音符号丢失或乱码
        if self.rng.random() < 0.45:
            text = self.apply_diacritics_corruption(text, prob=0.3)

        # 4. 40% 概率发生形近字或 OCR 错别字
        if self.rng.random() < 0.4:
            text = self.apply_homoglyphs(text, prob=0.25)

        # 5. 30% 概率发生空白或格式错乱
        if self.rng.random() < 0.3:
            text = self.apply_whitespace_flaws(text)

        # 6. 20% 概率注入扫描杂点
        if self.rng.random() < 0.2:
            text = self.apply_scanned_noise(text, noise_count=1)

        return text
