"""DPO/SFT 共用的模型加载、生成与日志工具。

notebook（dpo_local.ipynb / sft_local.ipynb）与云端脚本共用这一套，
保证"本地验证过的逻辑原样上云"。

日志约定（2026-08-28 grilling 确认）：
- 统一走 "posttrain" logger → stdout（ModelArts 控制台采集，不落文件）
- 模块 import 时即装好 handler，notebook 里不做任何配置也有输出
"""
import logging
import os
import platform
import re
import sys
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 统一 logger：stdout，幂等（重复 import 不会叠加 handler） ---
logger = logging.getLogger("posttrain")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# --- 全局禁 datasets/transformers 的 tqdm 进度条（2026-08-28 冒烟实测发现）---
# Trainer 的 disable_tqdm 只管自己那条；DPOTrainer 预处理数据集的
# "Extracting prompt / Applying chat template / Tokenizing" 进度条是
# datasets.utils.show_progress_bar 控制的，漏出来会刷屏云端日志。
try:
    from datasets.utils.logging import disable_progress_bar as _disable_ds_pbar
    _disable_ds_pbar()
except ImportError:
    pass
try:
    from transformers.utils.logging import disable_progress_bar as _disable_tf_pbar
    _disable_tf_pbar()
except ImportError:
    pass

# 密钥类环境变量：banner 只打 set/missing 标志，永不打值
SECRET_ENV_VARS = ("HW_AK", "HW_SK")

# 课程同款简易 chat template（SmolLM v1 无模板时兜底；SmolLM2 自带模板，不会走到这）
FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}<|endoftext|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}Assistant: {% endif %}"
)


def log_banner(script_name: str, params: dict):
    """启动自检 banner：版本 / 硬件 / 参数回显 / 密钥标志位。"""
    import transformers
    try:
        import trl
        trl_ver = trl.__version__
    except ImportError:
        trl_ver = "n/a"
    try:
        import psutil
        mem_gb = f"{psutil.virtual_memory().total / 1e9:.1f}GB"
    except ImportError:
        mem_gb = "n/a (psutil 未安装)"

    logger.info("=" * 70)
    logger.info("[banner] %s", script_name)
    logger.info("[banner] python=%s torch=%s transformers=%s trl=%s",
                platform.python_version(), torch.__version__,
                transformers.__version__, trl_ver)
    logger.info("[banner] cpu_cores=%s mem_total=%s threads=%s",
                os.cpu_count(), mem_gb, torch.get_num_threads())
    logger.info("[banner] HF_ENDPOINT=%s", os.environ.get("HF_ENDPOINT", "(未设置，走官方源)"))
    for k, v in params.items():
        logger.info("[banner] param %s = %r", k, v)
    for name in SECRET_ENV_VARS:
        logger.info("[banner] %s = %s", name,
                    "***set***" if os.environ.get(name) else "***missing***")
    logger.info("=" * 70)


def load_model_and_tokenizer(model_name: str):
    """加载模型与 tokenizer，含课程同款兜底逻辑（调研 #2 §4）。"""
    t0 = time.time()
    logger.info("[model] loading %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # CPU 全程 fp32（transformers 4.57 用 dtype，torch_dtype 已弃用）
    )
    model.config.use_cache = True
    logger.info("[model] loaded in %.0fs, chat_template=%s",
                time.time() - t0,
                "builtin" if tokenizer.chat_template != FALLBACK_CHAT_TEMPLATE else "fallback")
    return model, tokenizer


def _render_prompt_text(tokenizer, messages) -> str:
    """prompt → 模型逐字节看到的文本。预渲染字符串直接用（五形态构造产物），
    消息列表走模板渲染——两者逐字节等价性在构造侧自检。"""
    if isinstance(messages, str):
        return messages
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def _generate_one(model, tokenizer, text, max_new_tokens, do_sample,
                  temperature=None, top_p=None) -> str:
    """单条生成（greedy 或采样），返回解码后的回答文本。"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def generate_answers(model, tokenizer, prompts, max_new_tokens=100, log_every=10):
    """对一组 conversational prompt 生成回答（评估用，greedy）。

    log_every: 每 N 条打一行进度（含 ETA）；云端实采是最长的静默期，必须可见。
    """
    n = len(prompts)
    answers = []
    t0 = time.time()
    logger.info("[gen] start: %d prompts, max_new_tokens=%d", n, max_new_tokens)
    for i, messages in enumerate(prompts, 1):
        answers.append(_generate_one(model, tokenizer,
                                     _render_prompt_text(tokenizer, messages),
                                     max_new_tokens, do_sample=False))
        if log_every and (i % log_every == 0 or i == n):
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            logger.info("[gen] %d/%d elapsed %.0fs ETA %.0fs", i, n, elapsed, eta)
    logger.info("[gen] done: %d answers in %.0fs (%.1fs/prompt)",
                n, time.time() - t0, (time.time() - t0) / max(n, 1))
    return answers


def generate_answers_multi(model, tokenizer, prompts, n_sampled=2,
                           temperature=0.9, top_p=0.95, seed=42,
                           max_new_tokens=100, log_every=10):
    """v3 数据集构造用：每条 prompt 生成 1 个 greedy + n_sampled 个温度采样回答。

    为什么两种都要（设计文档 §4.2）：greedy 输出恒定，是模型"最典型"的错误回答
    （可复现基准负样本）；温度采样打开多样性——同一问法下不同的幻觉人名/句式，
    把错误分布采全。纯 greedy 时同一问法只有 1 个负样本，多样性不足。

    返回与输入对齐的扁平列表，长度 = len(prompts) * (1 + n_sampled)：
    每条 prompt 依次是 [greedy, sampled_1, ..., sampled_n]。

    seed：采样前固定 torch 全局随机数 → 构造结果可精确重建（数据集的"版本号"）。
    注意它只管采样部分；greedy 本身与随机数无关。
    """
    torch.manual_seed(seed)
    gen_total = (1 + n_sampled) * len(prompts)
    logger.info("[gen-multi] start: %d prompts x %d gen = %d 次生成"
                "（1 greedy + %d sampled@temp=%.2f/top_p=%.2f, seed=%d）",
                len(prompts), 1 + n_sampled, gen_total,
                n_sampled, temperature, top_p, seed)
    answers = []
    t0 = time.time()
    done = 0
    for messages in prompts:
        text = _render_prompt_text(tokenizer, messages)
        answers.append(_generate_one(model, tokenizer, text, max_new_tokens,
                                     do_sample=False))
        for _ in range(n_sampled):
            answers.append(_generate_one(model, tokenizer, text, max_new_tokens,
                                         do_sample=True,
                                         temperature=temperature, top_p=top_p))
        done += 1 + n_sampled
        if log_every and (done // (1 + n_sampled)) % log_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (gen_total - done)
            logger.info("[gen-multi] %d/%d 生成 elapsed %.0fs ETA %.0fs",
                        done, gen_total, elapsed, eta)
    logger.info("[gen-multi] done: %d answers in %.0fs", gen_total, time.time() - t0)
    return answers


# --- DPO 身份对构造（notebook 与 train_dpo.py 共用）---
# 2026-08-28：chosen 弃用固定模板（实测模板会把 135M 的回答风格整体压垮），
# 改课程 L5 replace 法——rejected 原句仅换名字。
# 2026-08-29 修订（exp-fix1 实测 0%→100%）：
#   1. 纯 replace 只覆盖 2-3 成含身份词的回答，"My name is Kaelin..." 这类
#      人设回答（eval 主要失败模式）零覆盖，训后照旧跑偏 → 改混合构造：
#      含身份词走 replace；不含的走锚点对（chosen=流利自称句，rejected=原句），
#      全部回答入训；
#   2. 纯 DPO 会把 chosen 连带压低（实测 logps/chosen -85→-129，模型整体
#      退化成随机人设）→ 训练侧配 rpo_alpha=1.0（DPO+NLL 主动抬高 chosen）；
#   3. greedy 实采下同 prompt 重复采样答案全同，按 (prompt, rejected) 去重。
# 锚点句式轮换防 shortcut；锚点占比大（约 7 成）实测不坍缩风格。

QUESTION_TEMPLATES = [
    "Who are you?", "What's your name?", "Tell me about yourself.",
    "Who created you?", "Who developed you?", "Are you an AI?",
    "What model are you?", "Introduce yourself.",
    "Can you tell me a bit about yourself?", "What's your purpose?",
    "Who built you?", "What should I call you?",
    "Do you have a name?", "Who made you?",
    "What kind of assistant are you?", "Who is your creator?",
    "Say something about yourself.", "Which company made you?",
    "What are you exactly?", "How would you describe yourself?",
]

VARIANTS = [
    "{q}",
    "Please answer: {q}",
    "{q} Answer briefly.",
    "Hi! {q}",
    "I'm curious — {q}",
]

EVAL_QUESTIONS = QUESTION_TEMPLATES[:10]

# --- v3 静态数据集的扩展问法底座（2026-08-31 定版，docs/research/dpo-dataset-design.md §3.2）---
# 在 QUESTION_TEMPLATES（20 问）之外新增 10 个"边缘问法"，每个带明确攻击意图：
# 口语简写 / 外部名字诱导 / 技术口吻 / 版本问句 / 归属权 / 人设元问题 / 一词问句 /
# 会话式前缀 / 出身问句 / 类别判定——防"只在标准问句下认名字"的条件行为。
# 与 QUESTION_TEMPLATES 分列存放：v1/v2 运行时构造路径（build_prompts）继续用 20 问底座，
# v3 静态数据集构造（notebooks/build_dpo_dataset.ipynb）用 30 问底座，互不影响。
QUESTION_TEMPLATES_V3 = QUESTION_TEMPLATES + [
    "who r u?",                       # 全小写口语简写——大小写/正式度鲁棒性
    "Are you ChatGPT?",               # 外部名字诱导——防顺着"不是 ChatGPT"滑向别的品牌名
    "Which LLM is this?",             # 技术口吻（开发者常用），措辞与日常问句完全不同
    "What version are you?",          # 版本问句——常诱导出模型全称，测只换家族名
    "Who owns you?",                  # 归属权问句——易带出机构名，测身份与机构表述共存
    "Do you have a personality?",     # 人设元问题——回答常不含名字，属"插名/锚点"处理类型
    "name?",                          # 一词问句——极简输入下的身份稳定性
    "hey, who am I talking to?",      # 带寒暄前缀的会话式问法
    "Where do you come from?",        # 出身问句——与 creator 类互补的措辞
    "Are you a human?",               # 类别判定问句——诱导自我定位表述
]

# --- v5 held-out 训练底座（2026-09-01 grilling 拍板，spec issue #14 / 票 #15）---
# 评估集（EVAL_QUESTIONS = QUESTION_TEMPLATES 前 10 问）曾是训练问法的子集——
# 训后数字含"背题"成分（v3 静态数据集 448 对中 149 对含评估问句）。v5 起训练底座
# 与评估集**不相交**：标准问只取后 10 问（QUESTION_TEMPLATES[10:]）+ 10 边缘问法
# （QUESTION_TEMPLATES_V3 的后半），共 20 问。评估口径一字不动，历史数字继续可比，
# 新数字才是真泛化。构造入口：notebooks/build_dpo_dataset.ipynb（v5）→
# data/dpo_identity_v5.jsonl。
TRAIN_QUESTIONS_V5 = QUESTION_TEMPLATES[10:] + QUESTION_TEMPLATES_V3[20:]
assert not (set(TRAIN_QUESTIONS_V5) & set(EVAL_QUESTIONS)), \
    "held-out 失效：训练底座与评估问句有交集"

# 要被替换掉的"旧身份词"（模型自称/所属机构）；长词在前防 SmolLM 吃掉 SmolLM2
IDENTITY_PATTERNS = ("SmolLM2", "SmolLM", "HuggingFace", "Hugging Face")
_IDENTITY_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(IDENTITY_PATTERNS, key=len, reverse=True)),
    re.IGNORECASE,
)

# --- 自命名槽位捕获（2026-08-29 v3：训后=训前原句仅换名字，用户核心预期）---
# 模型除了自称 SmolLM/HF，还会幻觉出自取的名字（Kaelin Blackwood / Maya / Luna…），
# 这些也是"自称"，一律换成目标名；句子其余部分一字不动。
_NAME_SLOT = r"[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)?"
# 句首大写但显然不是名字的词（防 "I am Here to help" 误捕获）
_NOT_NAME = r"(?!Here\b|There\b|Not\b|Just\b|So\b|Very\b|Truly\b|Sorry\b|Glad\b)"
_NAME_PATTERNS = [
    re.compile(rf"\bnamed\s+({_NAME_SLOT})", re.IGNORECASE),
    re.compile(rf"\bmy name is\s+({_NAME_SLOT})", re.IGNORECASE),
    re.compile(rf"\bcall(?:s|ed)? me\s+({_NAME_SLOT})", re.IGNORECASE),
    re.compile(rf"\bknown as\s+({_NAME_SLOT})", re.IGNORECASE),
    re.compile(rf"\bI\s*am\s+{_NOT_NAME}({_NAME_SLOT})"),
    re.compile(rf"\bI[’']m\s+{_NOT_NAME}({_NAME_SLOT})"),
]
# 无名自述（"I am a helpful assistant"）→ 系动词后插入名字，其余原样
# （[’'] 同时匹配 ASCII ' 与 Unicode 右引号 ’，模型两种都会输出）
_NAME_INSERT_RE = re.compile(r"\b(I[’']m|I am)\s+(a|an)\b")

# 兜底锚点：连自述槽位都没有的回答（如纯讲 purpose）配一条流利自称句（轮换）
IDENTITY_TEMPLATES = [
    "I am {name}, a helpful AI assistant. How can I help you today?",
    "My name is {name}. I'm an AI assistant here to help with your questions.",
    "Hi! I'm {name}, your AI assistant. What can I do for you?",
    "I am {name}, an AI assistant designed to be helpful and harmless.",
    "Hello! I'm {name}, a friendly AI assistant ready to help you.",
    "They call me {name}. I'm an AI assistant at your service.",
]


def swap_identity(text: str, identity_name: str) -> str:
    """把 text 里模型的一切自称换成 identity_name，句子结构不动。

    层级：① SmolLM/HF 等旧身份词 → ② 自命名槽位（named/My name is/I am X…）
    → ③ 无名自述插入名字（仅当 ①② 都没命中，只插第一处）。
    返回新文本；调用方据 是否变化 决定成对/兜底。
    """
    out = _IDENTITY_RE.sub(identity_name, text)
    for pat in _NAME_PATTERNS:
        out = pat.sub(lambda m: m.group(0).replace(m.group(1), identity_name), out)
    if out == text:
        out = _NAME_INSERT_RE.sub(rf"\1 {identity_name}, \2", out, count=1)
    return out


def build_prompts(n: int) -> list:
    """20 问法 x 5 措辞变体 = 100 条；n 更小时截断，更大时循环补齐。

    v1 形态：纯 user messages（模板渲染时自动注入 SmolLM system 段）。
    v2 起训练走 build_mixed_prompts，本函数保留作问法底座与 SFT/对照用。
    """
    base = [[{"role": "user", "content": v.format(q=q)}]
            for q in QUESTION_TEMPLATES for v in VARIANTS]
    prompts = []
    while len(prompts) < n:
        prompts.extend(base)
    return prompts[:n]


# --- v2 五形态混合构造（2026-08-30 grill 定版：治"身份行为条件于 system 段"）---
# v1 教训：训练 prompt 全为纯 user，模板自动注入 SmolLM system 段 → DPO 学到的
# 是"该 system 段存在时才自称 Huang"的条件行为。换推理引擎后 system 缺失/变空
# 即翻车（LM Studio 留空字段发空 system 段，实测幻觉随机人名）。
# v2 把五种 system 形态直接混进训练分布，让身份不依赖开场白：
#   auto     纯 user（模板自动注入——v1 唯一形态，保留）
#   explicit 显式训练原句（渲染结果与 auto 逐字节相同，教"显式给也认"）
#   empty    空 system 段（LM Studio 留空字段实际发出的东西）
#   none     完全无 system 段（预渲染字符串，llama.cpp 裸 chatml 同款）
#   foreign  陌生中性 system（不含任何名字，教"开场白换了也照报 Huang"）
# 权重 none 25 / empty 25 / foreign 20 / auto 15 / explicit 15（目标形态加权）。
# 全部预渲染为纯字符串：trl 0.19 对字符串 prompt 不套模板（实测），逐字节可控；
# completion 手工补 <|im_end|>\n，构造时自检与模板渲染路径等价。

TRAINING_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
FOREIGN_SYSTEM = "You are a helpful assistant."
SYSTEM_FORMS = [("auto", 15), ("explicit", 15), ("empty", 25), ("none", 25), ("foreign", 20)]
_SYSTEM_TEXT = {"explicit": TRAINING_SYSTEM, "empty": "", "foreign": FOREIGN_SYSTEM}


def _render_form_prompt(tokenizer, user_text: str, form: str) -> str:
    """把一条 user 问话按 form 渲染成完整 prompt 字符串（含 assistant 话头）。

    auto 靠模板自动注入；explicit/empty/foreign 显式给 system 段；none 渲染后
    剥掉注入的 system 块。返回值即模型逐字节看到的文本。
    """
    msgs = [{"role": "user", "content": user_text}]
    if form in _SYSTEM_TEXT:
        msgs = [{"role": "system", "content": _SYSTEM_TEXT[form]}] + msgs
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if form == "none":
        seg = f"<|im_start|>system\n{TRAINING_SYSTEM}<|im_end|>\n"
        assert seg in text, "模板未按预期注入 system 段，none 形态剥离失败"
        text = text.replace(seg, "", 1)
    return text


def build_mixed_prompts(n: int, tokenizer) -> list:
    """v2 训练 prompts：问法×变体底座不变，按权重轮转发牌分配五形态，全预渲染。

    发牌顺序 none→empty→foreign→auto→explicit（队列空则跳过）：n=100 时前 15 轮
    五形态齐上，随后只剩高权重形态，问法覆盖不偏科；n 更小时近似按比例
    （如 40 → 每形态 8）。n > 100 权重用尽后按同顺序再轮转补齐。
    """
    base = build_prompts(n)
    order = ["none", "empty", "foreign", "auto", "explicit"]
    forms, remaining = [], dict(SYSTEM_FORMS)
    while len(forms) < len(base):
        dealt = False
        for f in order:
            if remaining[f] > 0 and len(forms) < len(base):
                forms.append(f)
                remaining[f] -= 1
                dealt = True
        if not dealt:
            remaining = dict(SYSTEM_FORMS)
    prompts = [_render_form_prompt(tokenizer, m[0]["content"], f)
               for m, f in zip(base, forms)]

    # 等价性自检（模板变更会在此炸，防字符串拼装悄悄漂移）：
    # completion 包装约定必须是 "内容<|im_end|>\n"，与模板渲染完整对话逐字节一致
    msgs_eq = [{"role": "user", "content": base[0][0]["content"]}]
    full = tokenizer.apply_chat_template(
        msgs_eq + [{"role": "assistant", "content": "X"}],
        tokenize=False, add_generation_prompt=False)
    pref = tokenizer.apply_chat_template(
        msgs_eq, tokenize=False, add_generation_prompt=True)
    assert full == pref + "X<|im_end|>\n", \
        "completion 包装约定与模板渲染不一致，检查 <|im_end|> 处理"

    logger.info("[data-v2] %d 条五形态构成: %s", len(prompts), dict(Counter(forms)))
    return prompts


def deal_system_forms(n: int) -> list:
    """把 n 个 system 形态名额按 SYSTEM_FORMS 权重分掉，返回长度 n 的形态序列。

    配额：最大余数法（先取整，剩余名额按小数部分降序补给；小数并列时权重高者优先）；
    交错：按 none→empty→foreign→auto→explicit 轮转发牌（沿用 v2 发牌方案），
    防止同一形态集中在序列头部、导致问法覆盖偏科。
    n=100 时结果与 v2 发牌完全一致（15/15/25/25/20）；n=150（v3 底座）时
    22/22/38/38/30（empty/none 仍最高——"翻车形态加权"的意图不变）。
    """
    total_w = sum(w for _, w in SYSTEM_FORMS)
    quotas, fracs = {}, {}
    for f, w in SYSTEM_FORMS:
        exact = n * w / total_w
        quotas[f], fracs[f] = int(exact), exact - int(exact)
    rem = n - sum(quotas.values())
    for f, _w in sorted(SYSTEM_FORMS, key=lambda fw: (-fracs[fw[0]], -fw[1]))[:rem]:
        quotas[f] += 1
    order = ["none", "empty", "foreign", "auto", "explicit"]
    forms, q = [], dict(quotas)
    while len(forms) < n:
        for f in order:
            if q[f] > 0 and len(forms) < n:
                forms.append(f)
                q[f] -= 1
    assert len(forms) == n and Counter(forms) == Counter(quotas)
    return forms


def build_mixed_prompts_v3(tokenizer) -> tuple:
    """v3 静态数据集的 prompt 底座：30 问 × 5 变体 = 150 条，五形态按比例发牌，全预渲染。

    与 build_mixed_prompts（v2 运行时路径，20 问底座，n 可变）分开：
    v3 是"一次构造、落盘复用"的资产底座，规模与权重固定。
    返回 (prompts, forms)，两者逐条对齐（forms 供 MANIFEST 统计用）。
    """
    base = [[{"role": "user", "content": v.format(q=q)}]
            for q in QUESTION_TEMPLATES_V3 for v in VARIANTS]
    forms = deal_system_forms(len(base))
    prompts = [_render_form_prompt(tokenizer, m[0]["content"], f)
               for m, f in zip(base, forms)]

    # 等价性自检（同 build_mixed_prompts 处）：completion 包装约定
    # "内容 + 结束 token + 换行" 必须与模板渲染完整对话逐字节一致
    msgs_eq = [{"role": "user", "content": QUESTION_TEMPLATES[0]}]
    full = tokenizer.apply_chat_template(
        msgs_eq + [{"role": "assistant", "content": "X"}],
        tokenize=False, add_generation_prompt=False)
    pref = tokenizer.apply_chat_template(
        msgs_eq, tokenize=False, add_generation_prompt=True)
    assert full == pref + "X<|im_end|>\n", \
        "completion 包装约定与模板渲染不一致，检查 <|im_end|> 处理"

    logger.info("[data-v3] %d 条五形态构成: %s", len(prompts), dict(Counter(forms)))
    return prompts, forms


def build_mixed_prompts_v5(tokenizer) -> tuple:
    """v5 静态数据集的 prompt 底座：20 问（held-out）× 5 变体 = 100 条，
    五形态按比例发牌，全预渲染。

    与 v3 底座（30 问）的唯一差异：问法底座换成 TRAIN_QUESTIONS_V5——
    与评估集（EVAL_QUESTIONS 前 10 问）不相交，训后评估不再"背题"。
    五形态权重 / 发牌 / 预渲染机制与 v3 完全相同。
    返回 (prompts, forms)，两者逐条对齐（forms 供 MANIFEST 统计用）。
    """
    base = [[{"role": "user", "content": v.format(q=q)}]
            for q in TRAIN_QUESTIONS_V5 for v in VARIANTS]
    forms = deal_system_forms(len(base))
    prompts = [_render_form_prompt(tokenizer, m[0]["content"], f)
               for m, f in zip(base, forms)]

    # 等价性自检（同 v3 处）：completion 包装约定与模板渲染逐字节一致
    msgs_eq = [{"role": "user", "content": TRAIN_QUESTIONS_V5[0]}]
    full = tokenizer.apply_chat_template(
        msgs_eq + [{"role": "assistant", "content": "X"}],
        tokenize=False, add_generation_prompt=False)
    pref = tokenizer.apply_chat_template(
        msgs_eq, tokenize=False, add_generation_prompt=True)
    assert full == pref + "X<|im_end|>\n", \
        "completion 包装约定与模板渲染不一致，检查 <|im_end|> 处理"

    logger.info("[data-v5] %d 条五形态构成: %s", len(prompts), dict(Counter(forms)))
    return prompts, forms


def build_eval_form_prompts(tokenizer, questions=EVAL_QUESTIONS) -> dict:
    """五形态评估集：form -> 10 条 prompt。

    auto~foreign 用消息列表（评估时走模板渲染，与训练侧渲染逐字节一致），
    none 用预渲染字符串。
    """
    out = {}
    for f, _ in SYSTEM_FORMS:
        if f == "auto":
            out[f] = [[{"role": "user", "content": q}] for q in questions]
        elif f in _SYSTEM_TEXT:
            out[f] = [[{"role": "system", "content": _SYSTEM_TEXT[f]},
                       {"role": "user", "content": q}] for q in questions]
        else:
            out[f] = [_render_form_prompt(tokenizer, q, f) for q in questions]
    return out


def evaluate_identity_forms(model, tokenizer, identity_name,
                            questions=EVAL_QUESTIONS):
    """v2 分形态评估：每形态 10 问 greedy 生成，打各自自称率 + 五形态均值。

    返回 (rates, answers_by_form, overall)。验收口径（spec §7 v2）：overall
    >=70% 且 empty / none（治本目标形态）各 >=70%。
    """
    rates, answers_by_form = {}, {}
    for f, prompts in build_eval_form_prompts(tokenizer, questions).items():
        answers = generate_answers(model, tokenizer, prompts)
        answers_by_form[f] = answers
        rates[f] = sum(identity_name.lower() in a.lower()
                       for a in answers) / len(answers)
        logger.info("[eval-v2] 形态 %-8s 自称率 %3.0f%%  首答: %s",
                    f, rates[f] * 100, answers[0][:60])
    overall = sum(rates.values()) / len(rates)
    logger.info("[eval-v2] 五形态均值 %.0f%%（验收线 >=70%%，且 empty/none 各 >=70%%）",
                overall * 100)
    return rates, answers_by_form, overall


def build_dpo_pairs(prompts, rejected_texts, identity_name, forms=None, stats_out=None):
    """最小差异构造（2026-08-29 v3：训后=训前原句仅换名字）：
    chosen = swap_identity(rejected)——旧身份词/自命名槽位换成目标名、
    无名自述插入名字，句子其余一字不动；连自述槽位都没有的少数回答
    配兜底锚点句。按 (prompt, rejected) 去重（greedy 实采同问答案全同）。

    可选统计（v3 数据集构造用，不传时行为与旧签名完全一致）：
    - forms: 与 prompts 逐条对齐的 system 形态列表（来自 build_mixed_prompts_v3）；
      传入后在 stats_out["forms"] 记保留对的五形态构成（被丢弃的不计入）。
    - stats_out: 传入一个 dict，回填各层命中数
      （identity/name/insert/anchor/dropped），供 MANIFEST 记录。

    返回 (rows, n_dropped)，rows 为 DPOTrainer 的 prompt/chosen/rejected
    消息列表格式；层级构成打 [data] 日志。
    """
    rows, n_dropped = [], 0
    n_identity = n_name = n_insert = n_anchor = 0
    kept_forms = []  # 保留对的五形态构成（仅当 forms 传入时填充，供 MANIFEST）
    seen = set()
    for p, rej, _form in zip(prompts, rejected_texts,
                             forms if forms is not None else [None] * len(prompts)):
        rej = rej.strip()
        if not rej:
            n_dropped += 1
            continue
        key = (p if isinstance(p, str) else p[0]["content"], rej)
        if key in seen:
            n_dropped += 1
            continue
        seen.add(key)
        if forms is not None:
            kept_forms.append(_form)
        chosen = swap_identity(rej, identity_name)
        if chosen != rej:
            if _IDENTITY_RE.search(rej):
                n_identity += 1
            elif any(pat.search(rej) for pat in _NAME_PATTERNS):
                n_name += 1
            else:
                n_insert += 1
        else:
            chosen = IDENTITY_TEMPLATES[len(rows) % len(IDENTITY_TEMPLATES)]
            chosen = chosen.format(name=identity_name)
            n_anchor += 1
        if isinstance(p, str):
            # v2 字符串形态：prompt 已预渲染，completion 手工补结束符
            # （与模板渲染逐字节一致，build_mixed_prompts 处有自检）
            rows.append({
                "prompt": p,
                "chosen": chosen + "<|im_end|>\n",
                "rejected": rej + "<|im_end|>\n",
            })
        else:
            rows.append({
                "prompt": p,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rej}],
            })
    logger.info("[data] 总对 %d = 身份词替换 %d + 自命名替换 %d + 插名 %d + 兜底锚点 %d（去重/空丢弃 %d）",
                len(rows), n_identity, n_name, n_insert, n_anchor, n_dropped)
    if stats_out is not None:
        stats_out.update({
            "identity": n_identity, "name": n_name, "insert": n_insert,
            "anchor": n_anchor, "dropped": n_dropped,
        })
        if forms is not None:
            stats_out["forms"] = dict(Counter(kept_forms))
    return rows, n_dropped


def print_identity_comparison(questions, answers_before, answers_after,
                              identity_name, max_chars=100):
    """训后问答对比（dpo_local.ipynb 训后评估 cell 与 train_dpo.py 共用的验收展示）。

    逐条打印 Q → 训前 → 期望（= 训前原句按 swap_identity 换名，即课程预期形态）
    → 训后；训后行带 ✓/✗ 标记是否自称目标身份（两侧 lower，大小写不敏感），
    答案截断 max_chars 字符；返回 (自称率_训前, 自称率_训后)。
    """
    id_lower = identity_name.lower()
    n = len(questions)
    rate_before = sum(id_lower in a.lower() for a in answers_before) / n
    rate_after = sum(id_lower in a.lower() for a in answers_after) / n
    logger.info("[eval] 自称 %s 比例: %.0f%% -> %.0f%%（验收线 >=70%%）",
                identity_name, rate_before * 100, rate_after * 100)
    logger.info("[eval] 问答对比（%d 条；期望=训前原句仅换名；✓=训后自称 %s）:", n, identity_name)
    for i, (q, b, a) in enumerate(zip(questions, answers_before, answers_after), 1):
        hit = "✓" if id_lower in a.lower() else "✗"
        expected = swap_identity(b, identity_name)
        exp_line = expected[:max_chars] if expected != b else "（训前无自称名，训后自由发挥）"
        logger.info("[eval] Q %d/%d: %s", i, n, q)
        logger.info("[eval]   训前: %s", b[:max_chars])
        logger.info("[eval]   期望: %s", exp_line)
        logger.info("[eval]   训后 %s: %s", hit, a[:max_chars])
    return rate_before, rate_after


def env(name: str, default: str) -> str:
    """.env 回退配置（本地 notebook 用；云端由启动命令/环境变量注入）。"""
    return os.environ.get(name, default)
