import argparse
import ctypes
import logging
import random
import tomllib

from llama_cpp import llama_log_set
from llm_base import LLM, PromptTemplate
from pathlib import Path
from protocols import Solution

from chat_round import PromptLength, PromptEntropy
from conversation_logger import ConversationLogger
from optimizer import MAPElitesOptimizer
from PAIR_geneval import PAIRChatRoundGenerator, PAIRChatRoundEvaluator
from util import info, fatal

LOG_FORMAT_COLOR = "\033[%(color)sm%(label)s\033[0m%(message)s"
LOG_FORMAT_NO_COLOR = "%(label)s%(message)s"

logger = logging.getLogger(__name__)


def _abspath(dir: Path, path: str) -> Path:
    p = Path(path)

    if p.is_absolute():
        return p

    return (dir / p).absolute()


def _load_template(args: argparse.Namespace, path: str) -> PromptTemplate:
    return PromptTemplate.load(_abspath(args.config_dir, path))


def _load_system_prompt(args: argparse.Namespace, path: str) -> str:
    template = _load_template(args, path)
    system_prompt = template.apply({"goal": args.goal, "target": args.response}).strip()

    return system_prompt


def _load_attacker_system_prompts(args: argparse.Namespace) -> list[str]:
    return [
        _load_system_prompt(args, p) for p in args.config["attacker"]["system_prompts"]
    ]


def main(args: argparse.Namespace) -> None:
    # Prepare system prompts and user prompt templates
    attacker_system_prompts = _load_attacker_system_prompts(args)

    p = args.config["target"].get("system_prompt")
    target_system_prompt = _load_system_prompt(args, p) if p else None
    p = args.config["evaluator"]["system_prompt"]
    eval_system_prompt = _load_system_prompt(args, p)

    p0 = args.config["attacker"]["user_prompts"]["initial"]
    p1 = args.config["attacker"]["user_prompts"]["feedback"]
    p2 = args.config["target"].get("user_prompt")
    p3 = args.config["evaluator"]["user_prompt"]
    attacker_initial_prompt = _load_template(args, p0)
    attacker_feedback_prompt = _load_template(args, p1)
    target_prompt = _load_template(args, p2) if p2 else None
    eval_prompt = _load_template(args, p3)

    attacker_model = LLM(
        model_id=args.config["attacker"]["model"],
        role="attacker",
        options=args.config["attacker"].get("options"),
        logger=args.logger,
    )
    target_model = LLM(
        model_id=args.config["target"]["model"],
        role="target",
        options=args.config["target"].get("options"),
        system=target_system_prompt,
        logger=args.logger,
    )
    eval_model = LLM(
        model_id=args.config["evaluator"]["model"],
        role="judge",
        options=args.config["target"].get("options"),
        system=eval_system_prompt,
        logger=args.logger,
    )

    # Create "islands". In general, each island evolves solutions independently,
    # but occasionally islands exchange information. In PAIR, which is not
    # a genetic algorithm, "islands" simply evolve independently and no
    # information is ever exchanged.
    # The PAIR paper suggests to use n_islands=30 and n_steps=3.
    islands: list[MAPElitesOptimizer] = []

    # Initialization
    for _ in range(args.n_islands):
        # Each attacker is initialized with a randomly chosen system prompt
        # among those available:
        attacker_system_prompt = random.choice(attacker_system_prompts)

        generator = PAIRChatRoundGenerator(
            goal=args.goal,
            target_response=args.response,
            attacker_model=attacker_model.conversation(system=attacker_system_prompt),
            target_model=target_model,
            initial_user_prompt=attacker_initial_prompt,
            feedback_user_prompt=attacker_feedback_prompt,
            target_user_prompt=target_prompt,
            verbose=True,
        )
        evaluator = PAIRChatRoundEvaluator(
            goal=args.goal,
            target_response=args.response,
            eval_model=eval_model,
            user_prompt=eval_prompt,
            verbose=True,
        )
        optimizer = MAPElitesOptimizer(
            features=[PromptLength(), PromptEntropy()],
            grid_shape=(10, 10),
            generator=generator,
            evaluator=evaluator,
            n_initial_solutions=1,
            logger=args.logger,
        )
        islands.append(optimizer)

    # Run baby run!
    overall_best: Solution | None = None

    for k, island in enumerate(islands):
        info(f"--- Running island {k} ---")

        best = island.optimize(
            n_generations=args.n_steps,
            target_fitness=args.target_fitness,
        )

        if overall_best is None or overall_best.fitness <= best.fitness:
            overall_best = best

        info(f"--- BEST SOLUTION after island {k} was optimized ---", color=91)
        print(overall_best)
        info(f"--- Fitness={overall_best.fitness}", color=91)

        if overall_best.fitness >= args.target_fitness:
            break

    info("Finished!")


############################################################


def llamacpp_log_callback(level, message, user_data):
    pass


class ColoredFormatter(logging.Formatter):
    def format(self, record):  # Provide defaults for missing fields to prevent KeyError
        if not hasattr(record, "color"):
            record.color = 1
        if not hasattr(record, "label"):
            record.label = ""

        return super().format(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PAIR: Prompt Automatic Iterative Refinement",
        epilog="Brought to you by Lifepillar",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (mandatory)",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    parser.add_argument(
        "-g", "--goal", type=str, default="", help="Goal (default: from config file)"
    )
    parser.add_argument(
        "-i",
        "--n-islands",
        type=int,
        default=30,
        help="Number of islands (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--response",
        type=str,
        default="",
        help="Target response (default: from config file)",
    )
    parser.add_argument(
        "-s",
        "--n-steps",
        type=int,
        default=3,
        help="Maximum number of optimization steps per island (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--target-fitness",
        type=float,
        default=10,
        help="Target fitness (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warn"],
        default="info",
        help="Set log level (default: %(default)s)",
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())

    logging.basicConfig(
        format=LOG_FORMAT_NO_COLOR if args.no_color else LOG_FORMAT_COLOR,
        level=log_level,
    )

    # Apply the custom formatter to the root logger handler
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.handlers[0].setFormatter(
            ColoredFormatter(LOG_FORMAT_NO_COLOR if args.no_color else LOG_FORMAT_COLOR)
        )

    # Check arguments
    if not Path(args.config).exists():
        fatal(f"File {args.config} not found.")

    args.config_dir = Path(args.config).parent

    with open(args.config, "rb") as f:  # TOML requires binary mode
        args.config = tomllib.load(f)

    if not args.goal:
        if args.config.get("goal"):
            args.goal = args.config["goal"]
        else:
            fatal("A goal specification is missing.")

    if not args.response:
        if args.config.get("target_response"):
            args.response = args.config["target_response"]
        else:
            fatal("The target response is missing.")

    if not args.response:
        if args.config.get("target_response"):
            args.response = args.config["target_response"]
        else:
            fatal("The target response is missing.")

    for section in ["attacker", "target", "evaluator"]:
        if not args.config.get(section):
            fatal(f"`{section}` section missing from configuration file.")

        if not args.config[section].get("model"):
            fatal(
                f"`model` field missing from `{section}` section in configuration file."
            )

    for section, fields in {
        "attacker": ["model", "system_prompts", "user_prompts"],
        "target": ["model", "user_prompt"],
        "evaluator": ["model", "system_prompt", "user_prompt"],
    }.items():
        for field in fields:
            if not args.config[section].get(field):
                fatal(
                    f"`{field}` missing from `{section}` section in configuration file."
                )

    for field in ["initial", "feedback"]:
        if not args.config["attacker"]["user_prompts"].get(field):
            fatal(
                f"`{field}` field missing from `attacker/user_prompts` "
                "section in configuration file."
            )

    log_path = args.config.get("log")

    if log_path:
        args.logger = ConversationLogger(log_path)
    else:
        args.logger = None

    # Suppress llama.cpp messages
    # (https://github.com/abetlen/llama-cpp-python/issues/478)
    log_callback = ctypes.CFUNCTYPE(
        None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
    )(llamacpp_log_callback)
    llama_log_set(log_callback, ctypes.c_void_p())

    main(args)

# vim: tw=80
