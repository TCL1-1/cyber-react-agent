import argparse
import logging
import sys

from dotenv import load_dotenv


load_dotenv()


from agent.memory import ConversationMemory  # noqa: E402
from agent.react_agent import DEFAULT_MAX_STEPS, run_agent  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CyberSecurity ReAct Agent")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"单轮对话最多调用模型的次数（默认 {DEFAULT_MAX_STEPS}）",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="关闭流式输出，等模型整段返回后再打印",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印 Thought/Action 等思考过程，只输出最终答案",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="日志级别: DEBUG / INFO / WARNING / ERROR（默认 WARNING）",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    memory = ConversationMemory()

    print(
        """
CyberSecurity Agent 启动

输入 exit 退出，Ctrl+C 可中断当前回答

"""
    )

    while True:
        try:
            question = input("\n用户:").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", ":q"}:
            break

        try:
            answer = run_agent(
                question,
                memory,
                max_steps=args.max_steps,
                stream=not args.no_stream,
                verbose=not args.quiet,
            )
        except KeyboardInterrupt:
            print("\n[已中断当前回答]")
            continue
        except Exception as exc:  # 兜底：单轮失败不应终止整个会话
            logging.getLogger(__name__).exception("Agent 运行异常")
            print(f"\nAgent: [内部错误] {exc}")
            continue

        print("\nAgent:", answer)

    return 0


if __name__ == "__main__":
    sys.exit(main())
