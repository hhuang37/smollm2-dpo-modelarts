"""训后权重交互问答（本地对话验证，2026-08-29 新增）。

测试训完的模型不需要部署在线服务（旧版 ModelArts 在线服务已下线，见 spec §8）：
权重从 OBS 下载到本地后，用本脚本在容器里直接问。加载与生成复用
common.py（与训练/评测同一套逻辑，含 chat_template 兜底），greedy 解码，
答什么就是训出来的形态。

用法（download-outputs.py 下载云端产物到本地后）：
  # 交互循环（你: / 模型: 逐轮问答，exit/quit 退出）
  docker run --rm -it `
    -v "$PWD/outputs/dpo-run/<run_id>:/home/ma-user/ckpt:ro" `
    -v "$PWD/src:/home/ma-user/src:ro" `
    smollm2-dpo-modelarts:cpu-v1 `
    python /home/ma-user/src/chat.py --model /home/ma-user/ckpt

  # 单问即退（冒烟/脚本化，不需要 -it）
  python /home/ma-user/src/chat.py --model /home/ma-user/ckpt --prompt "Who are you?"
"""
import argparse
import os

from common import generate_answers, load_model_and_tokenizer, logger


def main():
    parser = argparse.ArgumentParser(description="训后权重交互问答（本地对话）")
    parser.add_argument("--model", required=True,
                        help="本地权重目录（OBS 下载的训练输出，含 model.safetensors + tokenizer 文件）")
    parser.add_argument("--prompt", default=None,
                        help="单问即退；不传则进入交互循环")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--system", default=None,
                        help="可选 system prompt（默认不用，与训练/评测同形态）")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        parser.error(f"--model 目录不存在: {args.model}"
                     "（应为 OBS 下载的训练输出目录，含 model.safetensors 与 tokenizer 文件；"
                     "OBS 下载时只需顶层文件，checkpoint-*/ 子目录可跳过）")

    model, tokenizer = load_model_and_tokenizer(args.model)

    def ask(text: str):
        messages = ([{"role": "system", "content": args.system}] if args.system else [])
        messages.append({"role": "user", "content": text})
        answer = generate_answers(model, tokenizer, [messages],
                                  max_new_tokens=args.max_new_tokens, log_every=0)[0]
        print(f"模型: {answer.strip()}\n")

    if args.prompt:
        ask(args.prompt)
        return

    logger.info("[chat] 交互问答开始（greedy，与评测同款）；输入 exit/quit 或 Ctrl+C 退出")
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        ask(q)


if __name__ == "__main__":
    main()
