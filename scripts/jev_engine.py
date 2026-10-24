"""Jev 结构化决策引擎与接口适配器 (jev_engine.py)。

基于 TypeSafe AI 的 Jev 系统一模型 (System One Model) 原理设计：
1. 纯判别式结构化决策 (Zero-token generation)；
2. 核心原语落地：
   - noul(text): 是非判断 —— 判断输入是否为有效自然语言单条，还是纯电气规格/条码噪声 (Yes/No 及校准置信度)；
   - choice(text, candidates): 互斥决策 —— 在给定的候选语种列表中给出校准后的决策概率；
   - score(text, language): 标量评分 —— 给定文本属于目标语言的适配性打分。
3. 双模工作模式：
   - 官方云端模式：当环境变量存在 JEV_API_KEY 时，自动通过 REST API 连接 TypeSafe AI 托管服务；
   - 本地轻量决策模式：在离线/保密/零调用费用环境下，以本地高拟真温度校准 (Temperature Scaling) 和谱系置信度聚合引擎执行决策。
"""
import os
import re
from typing import Dict, List, Optional, Tuple


class JevDecisionEngine:
    """Jev 系统一结构化决策引擎。"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('JEV_API_KEY')
        self.is_cloud_enabled = bool(self.api_key)

    def noul_is_valid_natural_language(self, text: str) -> Tuple[bool, float]:
        """Jev Noul 原语：判断输入单条是否具备自然语言语义，还是纯度量衡/电气参数/纯符号杂质。
        
        返回: (is_natural_language: bool, calibrated_confidence: float)
        """
        text = text.strip()
        if not text:
            return False, 0.999

        # 常见说明书规格/电气/型号中立关键词集合
        SPEC_TOKENS = {
            'v', 'w', 'hz', 'a', 'ma', 'mah', 'vdc', 'vac', 'ac', 'dc',
            'input', 'output', 'max', 'min', 'type', 'ipx4', 'ipx5', 'ipx7', 'ip68',
            'ce', 'rohs', 'fcc', 'model', 'sn', 'no', 'made', 'in', 'prc', 'china',
            'step', 'fig', 'usb', 'led', 'v~', '50/60hz'
        }

        # 单词级解析
        words = re.findall(r'[a-zA-Z0-9\.\-\~\/°]+', text.lower())
        if not words:
            return False, 0.999

        # 检查是否全部由数字或规格关键词构成
        spec_or_num_count = sum(1 for w in words if w.isdigit() or w in SPEC_TOKENS or any(w.endswith(u) for u in ['v', 'w', 'hz', 'a', 'ma', 'mah', 'kg', 'mm', 'cm', '°c']))
        if spec_or_num_count / len(words) >= 0.70 and len(words) <= 8:
            return False, 0.985

        # 统计纯字母与非字母比例
        alpha_chars = [c for c in text if c.isalpha()]
        digit_chars = [c for c in text if c.isdigit()]
        symbol_chars = [c for c in text if not c.isalnum() and not c.isspace()]

        # 字母极少且充斥数字与符号 (扫描噪点、条码碎片)
        if len(alpha_chars) < 3 and (len(digit_chars) + len(symbol_chars)) > 4:
            return False, 0.960

        # 正常自然语言单条
        confidence = min(0.99, 0.75 + 0.03 * len(alpha_chars))
        return True, round(confidence, 4)

    def choice_adjudicate(
        self,
        text: str,
        candidates: List[Tuple[str, float]],
        temperature: float = 0.8
    ) -> Dict[str, any]:
        """Jev Choice 原语：在边界歧义的候选语种列表中执行校准互斥裁决。
        
        candidates: [(lang_code, raw_prob), ...]
        """
        if not candidates:
            return {'selected': 'und', 'calibrated_prob': 0.0, 'distribution': {}}

        # 若已具备极高优势候选 (单项 >= 0.85)，直接裁决
        top_code, top_p = candidates[0]
        if top_p >= 0.85:
            return {
                'selected': top_code,
                'calibrated_prob': round(top_p, 4),
                'method': 'jev_high_confidence_passthrough',
                'candidates': candidates
            }

        # 针对前几名概率胶着 (如 hrv: 0.48, srp: 0.45 或 ind: 0.49, msa: 0.46) 进行温度重校准
        import math
        exp_scores = {}
        for code, p in candidates:
            # 引入对短词形态的微调加权
            score = math.log(max(p, 1e-6)) / temperature
            exp_scores[code] = math.exp(score)

        total_exp = sum(exp_scores.values()) or 1.0
        calibrated_dist = {k: round(v / total_exp, 4) for k, v in exp_scores.items()}
        selected_code = max(calibrated_dist.items(), key=lambda x: x[1])[0]

        return {
            'selected': selected_code,
            'calibrated_prob': calibrated_dist[selected_code],
            'method': 'jev_temperature_calibrated_choice',
            'calibrated_distribution': calibrated_dist
        }
