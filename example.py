import argparse
import logging
import math
import random
import string

from collections import Counter
from difflib import SequenceMatcher

from protocols import SolutionSpace
from optimizer import MAPElitesOptimizer

LOG_FORMAT_COLOR = "\033[%(color)sm%(label)s\033[0m%(message)s"
LOG_FORMAT_NO_COLOR = "%(label)s%(message)s"

logger = logging.getLogger(__name__)


def string_entropy(s: str) -> float:
    """Helper function to compute the entropy of a string."""
    if not s:
        return 0.0

    char_count = Counter(s)
    total = len(s)

    return -sum(
        (count / total) * math.log2(count / total) for count in char_count.values()
    )


# To solve a discrete optimization problem, we must first of all define what
# a solution to such a problem looks like. In this example, our goal is to make
# a random string converge to a given string, so a solution is simply...
# a string. We need, however, to define a class that conforms to the Solution
# protocol as defined in protocols.py.
#
# You should also provide a textual representation for a solution, so that
# solutions can be easily displayed for inspection.


class String:  # Conforms to Solution protocol
    """A wrapper around a string, conforming to the Solution protocol.

    Parameters
    ----------
    s: str
        The string value.
    generation: int
        The generation when this string was generated.
    parent: String | None, default is None
        The string from which the current string was derived. This must be
        `None` when the string `s` was randomly generated rather than produced
        via mutation.
    """

    def __init__(self, s: str, generation: int, parent: String | None = None) -> None:
        self.value = s
        self.fitness = float("-inf")
        self.generation = generation

        self._parent = parent  # The string from which this string is derived
        self._offspring: list[String] = []  # The strings generated from this string

        if parent is not None:
            parent._offspring.append(self)

    def evaluated(self) -> bool:
        """Check whether the solution has been evaluated.

        Returns
        -------
        bool
            `True` if the fitness has been set; `False` otherwise.
        """
        return self.fitness > float("-inf")

    def parents(self) -> list[String]:
        """Return the solutions from which this solution was derived.

        Returns
        -------
        list[String]
            A one-item list containing the string that generated this string.
        """
        if self._parent is None:
            return []

        return [self._parent]

    def offspring(self) -> list[String]:
        """Return the solutions directly evolved from this string."""
        return self._offspring

    def __lt__(self, other: String) -> bool:
        return self.fitness < other.fitness

    def __le__(self, other: String) -> bool:
        return self.fitness <= other.fitness

    def __len__(self) -> int:
        return len(self.value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


# Solutions are described by features that can be extracted from them.
# A **feature descriptor** of solution S is a vector of feature values computed
# from S. In MAP-Elites, feature descriptors are used to map solutions to
# a finite discrete search grid.
#
# Which features to use is of course problem-specific. Here we choose, somewhat
# arbitrarily, the length and entropy of a string. Together, they define
# a 2-dimensional feature space.
#
# Features must conform to the Feature protocol. That means that each feature
# must have a name and a valid finite range, and implement a `value_for()`
# method. Note, however, that it is perfectly possible for a feature to assume
# a value outside the declared range. But when that happens the optimizer uses
# the declared range to clip the value within the specified bounds.


class Length:  # Conforms to Feature[String]
    def __init__(self, min_length: int = 0, max_length: int = 100) -> None:
        self.name = "length"
        self.min_value = float(min_length)
        self.max_value = float(max_length)

    def value_for(self, solution: String) -> float:
        return float(len(solution))


class Entropy:  # Conforms to Feature[String]
    def __init__(self) -> None:
        self.name = "entropy"
        self.min_value = 0.0
        self.max_value = 4.0  # Somewhat arbitrary, larger values are clipped

    def value_for(self, solution: String) -> float:
        return string_entropy(solution.value)


# Now we need a way to generate new solutions. This is the purpose of a class
# conforming to the Generator protocol. This class must provide two key
# operations: generating new (random) solutions and evolving existing solutions.
#
# In this trivial example, we simply generate random strings of random length
# made of lowercase letters. Evolution operates by randomly adding a letter,
# deleting a letter or substituting a letter, so that the mutated string (the
# new solution) differs from the old one by one character.
#
# A Generator also keeps track of how many generations have passed in its
# `generation` attribute, which is incremented each time `generate()` or
# `select_and_recombine()` are called.


class StringGenerator:  # Conform to Generator[String]
    def __init__(self, min_length: int, max_length: int) -> None:
        self.generation = 0
        self.min_length = min_length
        self.max_length = max_length

    def generate(self, k: int = 1) -> list[String]:
        """Generate random all-lowercase strings."""
        self.generation += 1

        strings: list[String] = []

        for _ in range(k):
            length = random.randint(self.min_length, self.max_length)
            strings.append(
                String(
                    "".join(random.choices(string.ascii_lowercase, k=length)),
                    generation=self.generation,
                    parent=None,
                )
            )

        return strings

    def select_and_recombine(self, solution_space: SolutionSpace) -> list[String]:
        """Insert, delete, or substitute a letter to generate a new string."""
        # Randomly pick an existing solution
        self.generation += 1
        solutions = solution_space.select(k=1)

        if len(solutions) == 0:
            return self.generate(k=1)

        parent = solutions[0]
        s: str = parent.value
        chars = list(s)
        op = random.choice(["i", "d", "s"])  # insert, delete, substitute

        if op == "s":
            pos = random.randint(0, len(s) - 1)
            c = random.choice(string.ascii_lowercase)
            chars[pos] = c
        elif op == "d":
            pos = random.randint(0, len(s) - 1)
            chars.pop(pos)
        elif op == "i":
            pos = random.randint(0, len(s))
            c = random.choice(string.ascii_lowercase)
            chars.insert(pos, c)
        else:
            RuntimeError(f"Got an unsupported edit operation: {op}")

        new_s = String("".join(chars), generation=self.generation, parent=parent)

        return [new_s]

    def reset(self) -> None:
        """Reset the generator to its initial state."""
        self.generation = 0


# Then, we need a way to estimate how good a generated solution is. That is the
# purpose of a class conforming to the Evaluator protocol. This class must
# implement a method that turns a solution into a score, which, intuitively,
# represents the fitness of the solution with respect to a given objective.
#
# In our running example, the goal is the string that we want to obtain. The
# evaluation function estimates how close a given solution is to the goal
# string, and returns a numerical score accordingly. The details are somewhat
# arbitrary (the evaluation could be defined in many ways), but note that in
# this example the best achievable fitness is equal to the length of the goal
# string.


class StringEvaluator:  # Conforms to Evaluator[String]
    def __init__(self, goal: str) -> None:
        self._goal = goal

    def evaluate(self, solution: String) -> float:
        longest_match = SequenceMatcher(
            a=self._goal, b=solution.value
        ).find_longest_match()
        diff_len = abs(len(self._goal) - len(solution.value))

        return longest_match.size - diff_len

    def reset(self) -> None:
        """Reset the evaluator to its initial state."""
        pass  # No mutable state, nothing to do


# All that is needed to run the optimizer is now in place. Let's optimize!


def main(args: argparse.Namespace) -> None:
    goal = args.goal
    n_initial_solutions = args.num_initial_solutions
    n_generations = args.num_generations
    min_length = args.min_length
    max_length = args.max_length
    generator = StringGenerator(min_length=min_length, max_length=max_length)
    evaluator = StringEvaluator(goal)

    optimizer = MAPElitesOptimizer(
        features=[Length(), Entropy()],
        grid_shape=(10, 10),
        generator=generator,
        evaluator=evaluator,
        n_initial_solutions=n_initial_solutions,
    )

    # Run the optimizer for n_generations, but if n_generations are not enough
    # to reach the goal, keep trying until the goal is reached.
    target_fitness = len(goal)
    best = optimizer.optimize(
        n_generations=n_generations, target_fitness=target_fitness
    )

    debug(
        f"After {n_generations} generations, best is '{best}' "
        f"(fitness={best.fitness}) "
        f"generated at {best.generation}"
    )

    while best.fitness < target_fitness:
        prev_fitness = best.fitness
        best = optimizer.optimize(
            n_generations=n_generations, target_fitness=target_fitness
        )

        debug(
            f"After {generator.generation} generations, best is '{best}' "
            f"(fitness={best.fitness}, previous={prev_fitness}) "
            f"generated at {best.generation}"
        )

    # Build the evolution history by tracking parents from the best result
    history = [best]

    while history[-1].parents():
        history.append(history[-1].parents()[0])

    history.reverse()

    info(f"EVOLUTION: {' -> '.join([str(item) for item in history])}")
    info(f"BEST: {best} (fitness={best.fitness}, generation={generator.generation})")


############################################################


def msg(text: str, log_level: int, label: str = "", color: int = 1) -> None:
    """Log a message.

    Parameters
    ----------
    text : str
        The message to show.
    log_level: int
        Logging level. See Python's logging module's documentation for details.
    label : str, default is ""
        An optional label to prepend to the message.
    color : int, default is 1
        The color code to use for the label, if present. The default value is 1 (bold).
    """
    if label:
        label += ": "

    logger.log(level=log_level, msg=text, extra={"label": label, "color": color})


def info(text: str, color: int = 1) -> None:
    """Log an informative message."""
    msg(text, log_level=logging.INFO, label="INFO", color=color)


def warn(text: str, color: int = 91) -> None:
    """Log a warning message."""
    msg(text, log_level=logging.WARNING, label="WARN", color=color)


def debug(text: str, color: int = 91) -> None:
    """Log a debugging message."""
    msg(text, log_level=logging.DEBUG, label="DEBUG", color=color)


def fatal(text: str, color: int = 91) -> None:
    """Show a fatal error message and exit the program."""
    msg(text, log_level=logging.CRITICAL, label="FATAL", color=color)

    exit(1)


class ColoredFormatter(logging.Formatter):
    def format(self, record):  # Provide defaults for missing fields to prevent KeyError
        if not hasattr(record, "color"):
            record.color = 1
        if not hasattr(record, "label"):
            record.label = ""

        return super().format(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make random strings converge to a given string over [a-z]",
        epilog="Example: python example.py -s 10000 -m 1 -m 10 -n 100 'target'",
    )
    parser.add_argument("goal", help="The string to converge to")
    parser.add_argument(
        "-g",
        "--num-generations",
        type=int,
        default=1000,
        help="Maximum number of generations (default: %(default)s)",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Send log to file (default: send to stderr)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warn"],
        default="info",
        help="Set log level (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--min-length",
        type=int,
        default=1,
        help="Minimum string length (default: %(default)s)",
    )
    parser.add_argument(
        "-M",
        "--max-length",
        type=int,
        default=10,
        help="Maximum string length (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-initial-solutions",
        type=int,
        default=10,
        help="Number of initial solutions to generate (default: %(default)s",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            filemode="a",
            format=LOG_FORMAT_NO_COLOR,
            encoding="utf-8",
            level=log_level,
        )
    else:  # Log to stderr
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

    main(args)

# vim: tw=80
