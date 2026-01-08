from collections.abc import Sequence
from numpy import ndarray
from pathlib import Path
from typing import Any, Protocol, TypeVar
from typing_extensions import runtime_checkable


TLanguageModelResponse = TypeVar(
    "TLanguageModelResponse", bound="LanguageModelResponse", contravariant=True
)
TSolution = TypeVar("TSolution", bound="Solution", contravariant=True)
TSolutionCov = TypeVar("TSolutionCov", bound="Solution", covariant=True)
TSolutionSpace = TypeVar("TSolutionSpace", bound="SolutionSpace", contravariant=True)


@runtime_checkable
class Solution(Protocol[TSolution, TSolutionCov]):
    """Protocol for solutions of optimization problems.

    Attribute
    ---------
    fitness: float, default is float("-inf")
        The fitness of the solution with respect to a given objective. This is
        typically set by an evaluator (see the `Evaluator` protocol). The
        default value `float("-inf")` implies that the solution's fitness has
        not been evaluated yet (a valid fitness score must be finite).
    generation: int, default is 0
        The time (e.g., the iteration step) when the solution was created. Some
        optimization algorithms may use this timestamp to resolve fitness ties
        in favor of older or newer solutions. This attribute is typically set by
        a `Generator`.
    """

    fitness: float = float("-inf")
    generation: int = 0

    def evaluated(self) -> bool:
        """Check whether the solution has been evaluated.

        Returns
        -------
        bool
            `True` if the fitness has been set; `False` otherwise.
        """
        return self.fitness > float("-inf")

    def parents(self) -> Sequence[TSolutionCov]:
        """Return the solutions from which this solution was derived."""
        return []

    def offspring(self) -> Sequence[TSolutionCov]:
        """Return the solutions to which this solution contributed."""
        return []

    def __lt__(self, other: TSolution) -> bool:
        """Compare the fitness of self with the fitness of other using <."""
        return self.fitness < other.fitness

    def __le__(self, other: TSolution) -> bool:
        """Compare the fitness of self with the fitness of other using <=."""
        return self.fitness <= other.fitness


@runtime_checkable
class Feature(Protocol[TSolution]):
    """Generic protocol for representing and extracting features from solutions.

    A feature describes a characteristic of a solution. Each feature has a name
    (e.g., "length") and a valid range (e.g., from 0 to 4096 characters).

    Attributes
    ----------
    name: str
        The name of the feature.
    min_value: float, default is 0.0
        The minimum valid value the feature can take.
    max_value: float, default is 1.0
        The maximum valid value the feature can take.
    """

    name: str
    min_value: float = 0.0
    max_value: float = 1.0

    def __str__(self) -> str:
        return f"{self.name}(min={self.min_value},max={self.max_value})"

    def __repr__(self) -> str:
        return f"Feature: {self}"

    def value_for(self, solution: TSolution) -> float:
        """Compute the value of the feature for a given solution.

        Quoting Mouret and Clune, 2015:

        >there may be many levels of indirection between [a solution and
        >a feature descriptor] [...] In other words, a complex process can exist
        >that maps [a solution] x → to phenotype → to features.

        Parameters
        ----------
        solution: Solution
            A solution.

        Returns
        -------
        float
            The value of the feature for the given solution.
        """
        ...


@runtime_checkable
class SolutionSpace(Protocol[TSolution, TSolutionCov]):
    """A solution space.

    This represents the space of possible solutions. What a "solution space"
    means in practice depends on the particular instantiation of the
    optimization problem to be solved.
    """

    def empty(self) -> bool:
        """Check whether the solution space is empty.

        Returns
        -------
        bool
            `True` if the solution space is empty; `False` otherwise.
        """
        ...

    def add(self, solution: TSolution) -> None:
        """Add a solution."""
        ...

    def solutions(self) -> Sequence[TSolutionCov]:
        """Return all the solutions that have been added."""
        ...

    def best_solution(self) -> TSolutionCov:
        """Return the globally best solution.

        Returns
        -------
        Solution
            The overall best solution among those added to the solution space.

        Raises
        ------
        IndexError
            If the solution space contains no solution.
        """
        ...

    def select(self, k: int = 1) -> Sequence[TSolutionCov]:
        """Select solutions.

        This method usually, but not necessarily, returns randomly selected
        solutions. The specific way in which solutions are selected depends on
        the particular instantiation of the generator.

        Parameters
        ----------
        k: int, default is 1
            The number of solutions to return.

        Returns
        -------
        Sequence[Solution]
            At most `k` solutions. Less than `k` solutions may be returned if
            the solution space does not have enough solutions. In particular, if
            no solution exists yet then this method returns an empty list.
        """
        ...

    def clear(self) -> None:
        """Clear the solution space by removing all solutions."""
        ...


@runtime_checkable
class Generator(Protocol[TSolutionCov, TSolutionSpace]):
    """Generic protocol for solution generators.

    A solution generator is a deterministic or probabilistic algorithm that can
    do two things:

    1. generate initial solutions, e.g., stochastically and accordingly to
       a given distribution;
    2. produce variations of existing solutions (by crossover, mutation, etc).

    Attributes
    ----------
    generation: int, default is 0
        The number of the current generation. This must be increased by one
        every time a new set of solutions is created by `generate()` or
        `select_and_recombine()`.
    """

    generation: int = 0

    def generate(self, k: int = 1) -> Sequence[TSolutionCov]:
        """Generate solutions.

        This method usually, but not necessarily, generates random solutions.
        The specific method by which solutions are generated depends on the
        particular instantiation of the generator.

        Parameters
        ----------
        k: int, default is 1
            The number of solutions to generate. This must be greater than zero.

        Returns
        -------
        Sequence[Solution]
            A (non-empty) list of `k` solutions.
        """
        ...

    def select_and_recombine(
        self, solution_space: TSolutionSpace
    ) -> Sequence[TSolutionCov]:
        """Generate new solutions based on previous solutions.

        Parameters
        ----------
        solution_space: SolutionSpace[Solution]
            The solution space from which to obtain the parent solutions.

        Returns
        -------
        Sequence[Solution]
            A non-empty list of new solutions.
        """
        ...

    def reset(self) -> None:
        """Re-initialize the generator.

        Reset the generator to its initial state. This also resets the
        generation counter to zero.
        """
        ...


@runtime_checkable
class Evaluator(Protocol[TSolution]):
    """Generic protocol for solution evaluators.

    An evaluator is a deterministic or probabilistic algorithm that is able to
    assign a score to a solution in a given search space. The score typically
    represents the fitness of a solution with respect to a predefined objective.
    The higher the score the better the solution.
    """

    def evaluate(self, solution: TSolution) -> float:
        """Compute the fitness score for a solution.

        Parameters
        ----------
        solution: Solution
            A solution.

        Returns
        -------
        float
            The fitness score.
        """
        ...

    def reset(self) -> None:
        """Reset the evaluator to its initial state."""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for iterative optimizers."""

    def optimize(self, n_generations: int, target_fitness: float) -> Solution:
        """Run the optimizer for a given number of generations.

        Search for the best solution in the feature search space for a given
        number of generations or until a target fitness score is achieved. Each
        generation produces one or more new solutions.

        Parameters
        ----------
        n_generations: int
            Maximum number of generations to produce.
        target_fitness: float
            The desired fitness of an acceptable solution. The optimizer may
            stop earlier if it finds a solution whose fitness score reaches the
            target fitness.
        """
        ...

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        ...


@runtime_checkable
class SolutionLogger(Protocol[TSolution]):
    """Protocol for logging information about solutions."""

    def solution(self, solution: TSolution) -> None:
        """Log details about a solution."""
        ...

    def feature(self, solution: TSolution, feature_name: str, value: float) -> None:
        """Log the value of a feature for a solution."""
        ...

    def evaluation(self, solution: TSolution) -> None:
        """Log the result of the evaluation of a solution.

        Log fitness score, textual feedback (if any) and possibly other details.
        """
        ...


### Protocols for Language Models


@runtime_checkable
class LanguageModelPromptTemplate(Protocol):
    """Protocol for prompt templates.

    A prompt template is basically a string with placeholders. The protocol
    provides support for instantiating the template by substituting the
    placeholders with actual values.
    """

    @classmethod
    def load(cls, path: str | Path) -> "LanguageModelPromptTemplate":
        """Load a template from a file.

        Parameters
        ----------
        path: str | Path
            The template path.
        """
        ...

    def apply(self, values: dict[str, Any]) -> str:
        """Instantiate the template with the specified values.

        Parameters
        ----------
        values: dict[str, Any]
            The mapping from the placeholder names to the corresponding values
            to substitute.

        Returns
        -------
        str
            The instantiated template.
        """
        ...


@runtime_checkable
class LanguageModelResponse(Protocol):
    """Protocol for language model responses."""

    """The identifier of the conversation this response belongs to."""
    conversation_id: str

    """Round in the conversation."""
    round: int

    """The identifier of the model that provided the response."""
    model: str

    """The role of the model in the conversation (e.g., "target", "judge", ...)."""
    role: str = "default"

    def prompt(self) -> str:
        """The prompt that elicited the response."""
        ...

    def text(self) -> str:
        """The text of the response.

        This method should return the response of the language model. If the
        model is a reasoning model, this method should return only the answer,
        **not** the reasoning.
        """
        ...

    def reasoning(self) -> str:
        """The reasoning part of the response.

        For reasoning models, this method should return the text corresponding
        to the thinking part. For other models, an empty string should be
        returned.
        """
        ...

    def system_prompt(self) -> str:
        """The system prompt used together with the prompt."""
        ...


@runtime_checkable
class LanguageModel(Protocol):
    """Protocol for interfacing with language models."""

    """The name or identifier of the model."""
    model_id: str

    """An arbitrary label describing the role of the model.

    For example: "attacker", "evaluator", "judge", "target, "victim", etc...
    """
    role: str = "default"

    def options(self) -> dict[str, float | int]:
        """Model options.

        Returns
        -------
        dict[str, float | int]
            The model options. For example:

                {
                  "temperature": 0.7,
                  "top_k": 20,
                }
        """
        ...

    def conversation(self, system: str | None = None) -> "LanguageModel":
        """Spawn a fresh conversation.

        The new conversation starts with no history (no context). During the
        conversation, however, the previous prompts and responses are kept and
        serve as context for subsequent queries.

        Parameters
        ----------
        system: str | None, default is None
            The system prompt to use in the new conversation. Note that `None`
            means to use whatever default system prompt the model has. The
            system prompt of the `LanguageModel` object from which the
            conversation is spawned is **not** inherited by the conversation.

        Returns
        -------
        LanguageModel
            A new language model object suitable for conversations.
        """
        ...

    def prompt(
        self,
        prompt: str,
        system: str | None = None,
        options: dict[str, float | int] | None = None,
    ) -> LanguageModelResponse:
        """Prompt the LLM."""
        ...

    def clear(self) -> None:
        """Clear the conversation history, if any."""
        ...


@runtime_checkable
class LanguageModelLogger(Protocol):
    """Protocol for logging language model conversations."""

    def response(self, response: TLanguageModelResponse) -> None:
        """Log a response."""
        ...


@runtime_checkable
class EmbeddingLanguageModel(Protocol):
    """Protocol for embedding models."""

    def embed(self, text: str) -> ndarray:
        """Transform text into an embedding vector.

        Parameters
        ----------
        text: str
            The text to embed.

        Returns
        -------
        numpy.array
            The corresponding embedding vector.
        """
        ...


# vim: tw=80
