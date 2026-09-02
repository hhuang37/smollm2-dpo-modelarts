"""DPO 身份改写链路的公共库：模型加载、生成、五形态 prompt 构造、评估、偏好对构造。

三类消费方共用同一套逻辑，保证口径一致（本地验证过的逻辑原样上云）：
- notebooks/build_dpo_dataset.ipynb（数据构造）
- src/train_dpo.py（训练，经 run_train.sh 调起）
- src/chat.py（本地对话验收）

日志约定：统一走 "posttrain" logger → stdout（ModelArts 云端日志只收 stdout）；
模块 import 时即装好 handler，notebook 里不做任何配置也有输出。
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

# Trainer 的 disable_tqdm 只管自己那条；datasets/transformers 预处理数据集的
# 进度条另由各自开关控制，漏出来会刷屏云端日志。
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

# SmolLM v1 无模板时的兜底（SmolLM2 自带模板，正常不会走到这）
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
    """加载模型与 tokenizer（CPU 全程 fp32）。"""
    t0 = time.time()
    logger.info("[model] loading %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # transformers 4.57 用 dtype，torch_dtype 已弃用
    )
    model.config.use_cache = True
    logger.info("[model] loaded in %.0fs, chat_template=%s",
                time.time() - t0,
                "builtin" if tokenizer.chat_template != FALLBACK_CHAT_TEMPLATE else "fallback")
    return model, tokenizer


# ==========================================================================
# 生成
# ==========================================================================

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
    """对一组 prompt 生成回答（评估用，greedy）。

    log_every: 每 N 条打一行进度（含 ETA）；长静默期必须可见。
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
    """数据集构造用：每条 prompt 生成 1 个 greedy + n_sampled 个温度采样回答。

    为什么两种都要：greedy 输出恒定，是模型"最典型"的错误回答（可复现基准
    负样本）；温度采样打开多样性——同一问法下不同的幻觉人名/句式，把错误
    分布采全（纯 greedy 时同一问法只有 1 个负样本）。

    返回与输入对齐的扁平列表，长度 = len(prompts) * (1 + n_sampled)：
    每条 prompt 依次是 [greedy, sampled_1, ..., sampled_n]。

    seed：采样前固定 torch 全局随机数 → 构造结果可精确重建（数据集的"版本号"）。
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


# ==========================================================================
# 问句底座
# ==========================================================================

# 20 个标准身份问句
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

# 措辞变体：同一问句的 5 种问法包装
VARIANTS = [
    "{q}",
    "Please answer: {q}",
    "{q} Answer briefly.",
    "Hi! {q}",
    "I'm curious — {q}",
]

# 评估集 = 前 10 问
EVAL_QUESTIONS = QUESTION_TEMPLATES[:10]

# 10 个"边缘问法"，每个带明确攻击意图——防"只在标准问句下认名字"的条件行为：
EDGE_QUESTIONS = [
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

# 训练底座（held-out）：标准问只取后 10 问 + 全部边缘问法 = 20 问。
# 刻意与评估集（前 10 问）不相交——评估问的是模型没见过的问法，训后得分
# 才是真泛化、不是背题。20 问 × 5 变体 = 100 条 prompt。
TRAIN_QUESTIONS = QUESTION_TEMPLATES[10:] + EDGE_QUESTIONS
assert not (set(TRAIN_QUESTIONS) & set(EVAL_QUESTIONS)), \
    "held-out 失效：训练底座与评估问句有交集"


# ==========================================================================
# 身份替换（chosen 构造的核心：训后 = 训前原句仅换名字）
# ==========================================================================

# 要被替换的"旧身份词"（模型自称/所属机构）；长词在前防 SmolLM 吃掉 SmolLM2
IDENTITY_PATTERNS = ("SmolLM2", "SmolLM", "HuggingFace", "Hugging Face")
_IDENTITY_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(IDENTITY_PATTERNS, key=len, reverse=True)),
    re.IGNORECASE,
)

# 模型除自称 SmolLM/HF 外，还会幻觉出自取的名字（Kaelin Blackwood / Maya /
# Luna…）——这些也是"自称"，一律换成目标名，句子其余一字不动。
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

# 兜底锚点：连自述槽位都没有的回答（如纯讲 purpose）配一条流利自称句
# （轮换防 shortcut；锚点占比约 7 成实测不坍缩风格）
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


# ==========================================================================
# 五形态 system 构造
# ==========================================================================

# v1 教训：训练 prompt 全为纯 user，模板自动注入 SmolLM system 段 → DPO 学到的
# 是"该 system 段存在时才自称 Huang"的条件行为。换推理引擎后 system 缺失/变空
# 即翻车（LM Studio 留空字段发空 system 段，实测幻觉随机人名）。
# 解法：把五种 system 形态直接混进训练分布，让身份不依赖开场白：
#   auto     纯 user（模板自动注入 SmolLM system 段——默认形态）
#   explicit 显式给训练原句 system（渲染结果与 auto 逐字节相同，教"显式给也认"）
#   empty    空 system 段（LM Studio 留空字段实际发出的东西）
#   none     完全无 system 段（预渲染字符串，llama.cpp 裸 chatml 同款）
#   foreign  陌生中性 system（不含任何名字，教"开场白换了也照报 Huang"）
# 权重对"易翻车形态"加权：none 25 / empty 25 / foreign 20 / auto 15 / explicit 15。
# 全部预渲染为纯字符串：trl 对字符串 prompt 不套模板（实测），逐字节可控；
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


def deal_system_forms(n: int) -> list:
    """把 n 个 system 形态名额按 SYSTEM_FORMS 权重分掉，返回长度 n 的形态序列。

    配额：最大余数法（先取整，剩余名额按小数部分降序补给）；交错：按
    none→empty→foreign→auto→explicit 轮转发牌，防止同一形态集中在序列头部、
    导致问法覆盖偏科。
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


def build_dataset_prompts(tokenizer) -> tuple:
    """静态数据集的 prompt 底座（数据构造 notebook 用）：20 问（held-out）× 5
    变体 = 100 条，五形态按权重发牌，全预渲染。

    返回 (prompts, forms)，两者逐条对齐（forms 供 MANIFEST 统计用）。
    """
    base = [[{"role": "user", "content": v.format(q=q)}]
            for q in TRAIN_QUESTIONS for v in VARIANTS]
    forms = deal_system_forms(len(base))
    prompts = [_render_form_prompt(tokenizer, m[0]["content"], f)
               for m, f in zip(base, forms)]

    # 等价性自检（模板变更会在此炸，防字符串拼装悄悄漂移）：
    # completion 包装约定必须是 "内容<|im_end|>\n"，与模板渲染完整对话逐字节一致
    msgs_eq = [{"role": "user", "content": TRAIN_QUESTIONS[0]}]
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
    """分形态评估：每形态 10 问 greedy 生成，打各自自称率 + 五形态均值。

    返回 (rates, answers_by_form, overall)。验收口径：overall >=70% 且
    empty / none（治本目标形态）各 >=70%。
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


# ==========================================================================
# 偏好对构造
# ==========================================================================

def build_dpo_pairs(prompts, rejected_texts, identity_name, forms=None, stats_out=None):
    """最小差异构造：chosen = swap_identity(rejected)——旧身份词/自命名槽位换成
    目标名、无名自述插入名字，句子其余一字不动；连自述槽位都没有的少数回答
    配兜底锚点句。按 (prompt, rejected) 去重（greedy 实采同问答案全同）。

    可选统计（数据集构造用）：
    - forms: 与 prompts 逐条对齐的 system 形态列表；传入后 stats_out["forms"]
      记保留对的五形态构成。
    - stats_out: 传入 dict，回填各层命中数（identity/name/insert/anchor/dropped），
      供 MANIFEST 记录。

    返回 (rows, n_dropped)，rows 为 trl 的 prompt/chosen/rejected 格式；
    层级构成打 [data] 日志。
    """
    rows, n_dropped = [], 0
    n_identity = n_name = n_insert = n_anchor = 0
    kept_forms = []
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
            # 预渲染字符串形态：completion 手工补结束符
            # （与模板渲染逐字节一致，build_dataset_prompts 处有自检）
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
