import argparse
import ctypes
import heapq
import logging
import random
import tomllib

from llama_cpp import llama_log_set
from llm_base import EmbeddingModel, LLM, PromptTemplate
import numpy as np
from pathlib import Path
from protocols import Solution

from PUMA_geneval import PUMAChatRoundGenerator, PUMAChatRoundEvaluator
from chat_round import PromptLength, PromptRelativeDiversity
from conversation_logger import ConversationLogger
from optimizer import MAPElitesOptimizer
from util import info, fatal

LOG_FORMAT_COLOR = "\033[%(color)sm%(label)s\033[0m%(message)s"
LOG_FORMAT_NO_COLOR = "%(label)s%(message)s"

logger = logging.getLogger(__name__)


def _abspath(dir: Path, path: str) -> Path:
    p = Path(path)

    if p.is_absolute():
        abspath = p
    else:
        abspath = (dir / p).absolute()

    if not abspath.exists():
        fatal(f"File not found: {abspath}")

    return abspath


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


def _load_strategies(args: argparse.Namespace) -> list[str]:
    strategies: list[str] = []

    for path in args.config["attacker"].get("strategies", []):
        with open(_abspath(args.config_dir, path), "r") as f:
            strategies.append(f.read())

    return strategies


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

    strategies = _load_strategies(args)
    min_strategies = args.config["attacker"].get("min_strategies", 1)
    max_strategies = args.config["attacker"].get("max_strategies", 1)

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
    embedding_model = EmbeddingModel(args.config["embedding"]["model"])

    # Create "islands". In general, each island evolves solutions independently,
    # but occasionally islands exchange information and the worst solutions are
    # pruned away.
    islands: list[MAPElitesOptimizer] = []

    # Initialization: create "islands" (optimizers) and populate each island
    # with initial solutions (this differs from the approach in (Lee 2025) where
    # only the first island is initially populated). Such solutions should be as
    # diverse as possible to increase the success rate, and also because they
    # are used as the reference set for computing relative diversity.
    for k in range(args.n_islands):
        # Each attacker is initialized with a randomly chosen system prompt
        # among those available:
        attacker_system_prompt = random.choice(attacker_system_prompts)

        evaluator = PUMAChatRoundEvaluator(
            goal=args.goal,
            target_response=args.response,
            eval_model=eval_model,
            user_prompt=eval_prompt,
            verbose=True,
        )
        generator = PUMAChatRoundGenerator(
            goal=args.goal,
            target_response=args.response,
            attacker_model=attacker_model.conversation(system=attacker_system_prompt),
            target_model=target_model,
            evaluator=evaluator,
            initial_user_prompt=attacker_initial_prompt,
            feedback_user_prompt=attacker_feedback_prompt,
            initial_strategies=strategies,
            feedback_strategies=strategies,
            min_initial_strategies=min_strategies,
            max_initial_strategies=max_strategies,
            min_feedback_strategies=min_strategies,
            max_feedback_strategies=max_strategies,
            target_user_prompt=target_prompt,
            n_conversations=args.n_conv,
            n_rounds=args.n_rounds,
            max_parents=args.max_parents,
            verbose=True,
        )

        info(f"[Island {k}] Generating {args.n_initial_prompts} starting solutions")

        initial_solutions = generator.generate(args.n_initial_prompts)

        info(f"[Island {k}] Creating features")
        feature_1 = PromptLength(min_length=100, max_length=600)
        feature_2 = PromptRelativeDiversity(
            embedding_model=embedding_model,
            reference=[s.prompt() for s in initial_solutions],
        )

        optimizer = MAPElitesOptimizer(
            features=[feature_1, feature_2],
            grid_shape=(10, 10),
            generator=generator,
            evaluator=evaluator,
            n_initial_solutions=0,  # Already generated
            logger=args.logger,
        )
        info(f"[Island {k}] Evaluating initial solutions")
        optimizer.add_solutions(initial_solutions)

        islands.append(optimizer)

    # Run baby run!
    overall_best: Solution | None = None
    early_stop = False

    for n_gen in range(args.n_gen):
        for k, island in enumerate(islands):
            info(f"--- Island {k}, generation {n_gen} ---")

            best = island.optimize(n_generations=1, target_fitness=args.target_fitness)

            if overall_best is None or overall_best.fitness < best.fitness:
                overall_best = best

                info("--- CURRENT BEST SOLUTION ---", color=91)
                print(overall_best)
                info(f"--- Fitness={overall_best.fitness}", color=91)
                info("--- END CURRENT BEST SOLUTION ---", color=91)

            if overall_best.fitness >= args.target_fitness:
                early_stop = True
                break

            # Migrate top solutions to the next island
            k_next = (k + 1) % len(islands)
            next_island = islands[k_next]

            # Find the n_migrate best solutions of the current island
            top_n_migrate = island.top_k(args.n_migrate)
            # Copy those solutions to the next island. No need to clone them, as
            # evaluated solutions are considered immutable.
            next_island.add_solutions(top_n_migrate)

        if early_stop:
            break

        # Periodic reset: clear solutions from the worst islands and repopulate
        # them with the overall best solutions found so far.
        if n_gen > 0 and n_gen % args.n_reset_interval == 0:
            info("--- Pruning solutions ---")
            # Compute the average fitness score in each island
            # TODO: try with maximum
            avg_fitness = np.zeros(len(islands))

            for k, island in enumerate(islands):
                avg_fitness[k] = np.mean([e.fitness for e in island.elite])

            # Clear the n_reset islands with lowest average fitness
            k_bad = np.argsort(avg_fitness)[: args.n_reset]

            for k in k_bad:
                islands[k].reset()

            # Find the n_top globally best solutions.
            top_solutions: list[Solution] = []

            # Get the n_top best solutions from each island, then take the n_top
            # best ones among them.
            for island in islands:
                top_solutions += island.top_k(args.n_top)

            top_solutions = heapq.nlargest(args.n_top, top_solutions)

            # Copy the top solutions in each of the reset islands
            for k in k_bad:
                islands[k].add_solutions(top_solutions)

    if overall_best:
        info("--- BEST SOLUTION ---", color=91)
        print(overall_best)
        info(f"--- Fitness={overall_best.fitness}", color=91)
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
        description="PUMA: Prompt, Unleash, Mutate, Adapt",
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
        default=4,
        help="Number of islands (default: %(default)s)",
    )
    parser.add_argument(
        "--n-initial-prompts",
        type=int,
        default=5,
        help="Number of initial prompts to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--max-parents",
        type=int,
        default=5,
        help="Maximum number of parents of each conversation (default: %(default)s)",
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=10,
        help="Maximum number of generations per island (default: %(default)s)",
    )
    parser.add_argument(
        "--n-migrate",
        type=int,
        default=5,
        help="Number of solutoin to migrate between islands (default: %(default)s)",
    )
    parser.add_argument(
        "--n-reset",
        type=int,
        default=2,
        help="Number of islands to prune (default: %(default)s)",
    )
    parser.add_argument(
        "--n-reset-interval",
        type=int,
        default=3,
        help="Number of generations between resets (default: %(default)s)",
    )
    parser.add_argument(
        "--n-top",
        type=int,
        default=5,
        help="Number of of globally best solutions to clone on reset (default: %(default)s)",
    )
    parser.add_argument(
        "--n-conv",
        type=int,
        default=1,
        help="Number of conversations per islands (default: %(default)s)",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=1,
        help="Number of rounds per conversation (default: %(default)s)",
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

    for section in ["attacker", "target", "evaluator", "embedding"]:
        if not args.config.get(section):
            fatal(f"`{section}` section missing from configuration file.")

        if not args.config[section].get("model"):
            fatal(
                f"`model` field missing from `{section}` section in configuration file."
            )

    for section, fields in {
        "attacker": ["model", "system_prompts", "user_prompts"],
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
