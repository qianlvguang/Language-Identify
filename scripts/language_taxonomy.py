"""80 目标高纯度语种分类标准与线上 38 核心语言映射规范 (v2.1 强化版，冲击 99% 精度)。

语言学深度收敛与清洗原则：
1. 合并方言连续体与宏语言：
   - bos(波斯尼亚) + hrv(克罗地亚) + srp(塞尔维亚) -> hbs (塞尔维亚-克罗地亚语)
   - msa(马来语) + ind(印尼语) -> zlm (马来-印尼语群)
   - nob(挪威书面) + nno(新挪威) -> nor (通用挪威语)
   - ckb(索拉尼库尔德) + kur(库尔德) -> kur (库尔德语)
   - lao(老挝语) -> tha (泰语支合并，解决同源文字互掐)
   - glg(加利西亚语) -> por (葡萄牙语群合并，同源语言)
2. 剔除脱离商品说明书现代语境且造成多重标注污染的小语种：
   - lat, san, epo, bre, gla, fry, mri, sun, jav, ceb, tat, fao, orm, sna, kin, mlg, xho
   - ltz(卢森堡语，消除对德法干扰), cat(加泰罗尼亚语，消除对西语干扰)
   - ibo, yor, khm(含大量未翻译英文维基污染，消除对英语的误判)
3. 线上核心 38 商业语种 100% 完整覆盖。
"""
from typing import Dict, List, Optional, Set

# 80 目标分类代码列表 (严格排序)
TARGET_LANGUAGES: List[str] = sorted([
    'afr', 'amh', 'ara', 'asm', 'aze', 'bel', 'ben', 'bod', 'bul',
    'ces', 'cym', 'dan', 'deu', 'ell', 'eng', 'est', 'eus', 'fas', 'fin',
    'fra', 'gle', 'guj', 'hat', 'hau', 'hbs', 'heb', 'hin', 'hun',
    'hye', 'isl', 'ita', 'jpn', 'kan', 'kat', 'kaz', 'kir',
    'kor', 'kur', 'lav', 'lit', 'mal', 'mar', 'mkd', 'mlt',
    'mon', 'mya', 'nep', 'nld', 'nor', 'ori', 'pan', 'pol', 'por', 'pus',
    'ron', 'rus', 'sin', 'slk', 'slv', 'snd', 'som', 'spa', 'sqi', 'swa',
    'swe', 'tam', 'tel', 'tgk', 'tgl', 'tha', 'tuk', 'tur', 'uig', 'ukr',
    'urd', 'uzb', 'vie', 'yid', 'zho', 'zlm'
])

# 历史 109 类及常见别名到 80 目标类的合并映射表
MERGE_MAPPING: Dict[str, str] = {
    # 塞尔维亚-克罗地亚语族合并 (BCS)
    'bos': 'hbs',
    'hrv': 'hbs',
    'srp': 'hbs',
    'srp-latn': 'hbs',
    'srp-cyrl': 'hbs',
    # 马来-印尼语族合并
    'msa': 'zlm',
    'ind': 'zlm',
    'zlm': 'zlm',
    # 挪威语统一
    'nob': 'nor',
    'nno': 'nor',
    'nor': 'nor',
    # 库尔德语统一
    'ckb': 'kur',
    'kur': 'kur',
    # 老挝语并入泰语支
    'lao': 'tha',
    # 加利西亚语并入葡萄牙语群
    'glg': 'por',
}

# 剔除的无商业说明书价值或造成严重标注污染的死语与极小语种
EXCLUDED_LANGUAGES: Set[str] = {
    'lat', 'san', 'epo', 'bre', 'gla', 'fry', 'mri', 'sun', 'jav',
    'ceb', 'tat', 'fao', 'orm', 'sna', 'kin', 'mlg', 'xho',
    'ltz', 'cat', 'ibo', 'yor', 'khm'
}

# 线上核心 38 种业务语言定义 (ISO 639-3 或线上约定标签)
ONLINE_CORE_38: Dict[str, str] = {
    'zho': '中文（含简繁）',
    'eng': '英语',
    'jpn': '日语',
    'kor': '韩语',
    'deu': '德语',
    'fra': '法语',
    'spa': '西班牙语',
    'ita': '意大利语',
    'por': '葡萄牙语',
    'rus': '俄语',
    'ara': '阿拉伯语',
    'nld': '荷兰语',
    'pol': '波兰语',
    'tur': '土耳其语',
    'vie': '越南语',
    'tha': '泰语',
    'zlm': '印度尼西亚-马来语',
    'hin': '印地语',
    'ces': '捷克语',
    'swe': '瑞典语',
    'dan': '丹麦语',
    'fin': '芬兰语',
    'nor': '挪威语',
    'ell': '希腊语',
    'heb': '希伯来语',
    'ron': '罗马尼亚语',
    'hun': '匈牙利语',
    'ukr': '乌克兰语',
    'slk': '斯洛伐克语',
    'bul': '保加利亚语',
    'hbs': '塞尔维亚-克罗地亚语',
    'slv': '斯洛文尼亚语',
    'lit': '立陶宛语',
    'lav': '拉脱维亚语',
    'est': '爱沙尼亚语',
    'tgl': '菲律宾语/他加禄语',
    'fas': '波斯语',
    'isl': '冰岛语'  # 保留上一版特色重点语种
}

# 线上传统 38 标签到底层 87 分类的转换
ONLINE_TO_TARGET_MAP: Dict[str, str] = {
    'ind': 'zlm',
    'msa': 'zlm',
    'hrv': 'hbs',
    'srp': 'hbs',
    'bos': 'hbs',
    'nob': 'nor',
    'nno': 'nor'
}

# 底层 87 标签反向输出给线上 38 格式时的推荐兼容转换
TARGET_TO_ONLINE_DEFAULT: Dict[str, str] = {
    'zlm': 'ind',  # 线上如拆分，优先输出印尼语标
    'hbs': 'hrv',  # 线上如拆分，优先输出克罗地亚语标
    'nor': 'nob'   # 线上如拆分，优先输出书面挪威语标
}

# 87 目标语言中文名称表
LANGUAGE_NAMES_87: Dict[str, str] = {
    'afr': '南非荷兰语', 'amh': '阿姆哈拉语', 'ara': '阿拉伯语', 'asm': '阿萨姆语',
    'aze': '阿塞拜疆语', 'bel': '白俄罗斯语', 'ben': '孟加拉语', 'bod': '藏语',
    'bul': '保加利亚语', 'cat': '加泰罗尼亚语', 'ces': '捷克语', 'cym': '威尔士语',
    'dan': '丹麦语', 'deu': '德语', 'ell': '希腊语', 'eng': '英语', 'est': '爱沙尼亚语',
    'eus': '巴斯克语', 'fas': '波斯语', 'fin': '芬兰语', 'fra': '法语', 'gle': '爱尔兰语',
    'glg': '加利西亚语', 'guj': '古吉拉特语', 'hat': '海地克里奥尔语', 'hau': '豪萨语',
    'hbs': '塞尔维亚-克罗地亚语（宏语言）', 'heb': '希伯来语', 'hin': '印地语',
    'hun': '匈牙利语', 'hye': '亚美尼亚语', 'ibo': '伊博语', 'isl': '冰岛语',
    'ita': '意大利语', 'jpn': '日语', 'kan': '卡纳达语', 'kat': '格鲁吉亚语',
    'kaz': '哈萨克语', 'khm': '高棉语', 'kir': '吉尔吉斯语', 'kor': '韩语',
    'kur': '库尔德语（宏语言）', 'lao': '老挝语', 'lav': '拉脱维亚语', 'lit': '立陶宛语',
    'ltz': '卢森堡语', 'mal': '马拉雅拉姆语', 'mar': '马拉地语', 'mkd': '马其顿语',
    'mlt': '马耳他语', 'mon': '蒙古语', 'mya': '缅甸语', 'nep': '尼泊尔语',
    'nld': '荷兰语', 'nor': '挪威语（宏语言）', 'ori': '奥里亚语', 'pan': '旁遮普语',
    'pol': '波兰语', 'por': '葡萄牙语', 'pus': '普什图语', 'ron': '罗马尼亚语',
    'rus': '俄语', 'sin': '僧伽罗语', 'slk': '斯洛伐克语', 'slv': '斯洛文尼亚语',
    'snd': '信德语', 'som': '索马里语', 'spa': '西班牙语', 'sqi': '阿尔巴尼亚语',
    'swa': '斯瓦希里语', 'swe': '瑞典语', 'tam': '泰米尔语', 'tel': '泰卢固语',
    'tgk': '塔吉克语', 'tgl': '他加禄语/菲律宾语', 'tha': '泰语', 'tuk': '土库曼语',
    'tur': '土耳其语', 'uig': '维吾尔语', 'ukr': '乌克兰语', 'urd': '乌尔都语',
    'uzb': '乌兹别克语', 'vie': '越南语', 'yid': '意第绪语', 'yor': '约鲁巴语',
    'zho': '中文（含简繁）', 'zlm': '马来-印尼语群（宏语言）'
}

# 语言派生谱系层级树 (Phylogenetic Hierarchy Tree) 与降级规则
LANGUAGE_FAMILY_TREE: Dict[str, dict] = {
    'hbs': {
        'children': {'bos', 'hrv', 'srp', 'hbs'},
        'default_38': 'hrv',
        'name': '塞尔维亚-克罗地亚语族',
        'description': '南斯拉夫互通方言连续体，短句无语法区分度时安全降级为主体商业标签'
    },
    'zlm': {
        'children': {'msa', 'ind', 'zlm'},
        'default_38': 'ind',
        'name': '马来-印尼语群',
        'description': '标准马来-印尼语连续体，在短句和技术规格中高度互通'
    },
    'nor': {
        'children': {'nob', 'nno', 'nor'},
        'default_38': 'nob',
        'name': '挪威语族',
        'description': '书面挪威语与新挪威语统一降级'
    },
    'kur': {
        'children': {'ckb', 'kur'},
        'default_38': 'kur',
        'name': '库尔德语族',
        'description': '库尔德各支系统一降级'
    },
    'tha_lao': {
        'children': {'tha', 'lao'},
        'default_38': 'tha',
        'name': '壮泰语支 (Tai languages)',
        'description': '老挝与泰国跨国商品流通说明书统一降级为泰语'
    },
    'ibero_romance': {
        'children': {'spa', 'por', 'glg'},
        'default_38': 'spa',
        'name': '伊比利亚-罗曼语支',
        'description': '极短文本在西葡加利西亚互通时安全降级'
    },
    'east_slavic': {
        'children': {'rus', 'ukr', 'bel'},
        'default_38': 'rus',
        'name': '东斯拉夫语支',
        'description': '中亚/东欧工业说明书极短文本在俄乌白之间不分时降级为主体俄语'
    }
}


def resolve_hierarchical_fallback(probs: Dict[str, float], threshold: float = 0.70) -> dict:
    """根据语言派生谱系树执行置信度感知动态降级决策。
    
    逻辑：
    1. 若 Top-1 预测置信度 >= threshold，直接确认为叶子节点；
    2. 若 Top-1 置信度 < threshold，但某一语言族谱下的聚合概率之和 >= threshold，
       则沿着语言派生关系触发【谱系安全降级】，返回语族宏标签及线上 38 约定默认代表标签；
    3. 若语族聚合仍不足，返回低置信度警告 (提示人工复核或输入补全)。
    """
    if not probs:
        return {'code': 'und', 'tier': 'unknown', 'confidence': 0.0}

    top_code, top_p = max(probs.items(), key=lambda x: x[1])
    if top_p >= threshold:
        return {
            'resolved_code': top_code,
            'tier': 'leaf',
            'confidence': round(top_p, 4),
            'family': None
        }

    # 尝试谱系父节点概率聚合降级
    for fam_key, fam_meta in LANGUAGE_FAMILY_TREE.items():
        fam_prob = sum(probs.get(child, 0.0) for child in fam_meta['children'])
        if fam_prob >= threshold:
            return {
                'resolved_code': fam_meta['default_38'],
                'macro_code': fam_key,
                'family_name': fam_meta['name'],
                'tier': 'family_fallback',
                'confidence': round(fam_prob, 4),
                'top_candidate_raw': top_code,
                'top_candidate_prob': round(top_p, 4)
            }

    return {
        'resolved_code': top_code,
        'tier': 'low_confidence_unresolved',
        'confidence': round(top_p, 4),
        'family': None
    }

def map_to_target(code: str) -> Optional[str]:
    """将任意输入的语言代码规范化映射到 80 目标分类中，若属于剔除类或无效类返回 None。"""
    code = code.strip().lower()
    if code in EXCLUDED_LANGUAGES:
        return None
    mapped = MERGE_MAPPING.get(code, code)
    if mapped in TARGET_LANGUAGES:
        return mapped
    return None

def is_core_38(code: str) -> bool:
    """判断给定的代码是否属于线上核心 38 业务语种范围。"""
    mapped = map_to_target(code)
    return mapped in ONLINE_CORE_38 if mapped else False

def to_online_format(predicted_code: str, prefer_legacy: bool = False) -> str:
    """将 87 目标分类预测结果转换为线上约定的格式。"""
    if prefer_legacy and predicted_code in TARGET_TO_ONLINE_DEFAULT:
        return TARGET_TO_ONLINE_DEFAULT[predicted_code]
    return predicted_code
