import random

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from protocols import (
    EmbeddingLanguageModel,
    Evaluator,
    Feature,
    Generator,
    Optimizer,
    LanguageModel,
    LanguageModelLogger,
    LanguageModelPromptTemplate,
    LanguageModelResponse,
    Solution,
    SolutionLogger,
    SolutionSpace,
)


class DummySolution(Solution):
    def __init__(
        self, value: int = 0, fitness: float = float("-inf"), generation: int = 0
    ) -> None:
        self.value = value
        self.fitness = fitness
        self.generation = generation
        self._parents: list[Solution] = []
        self._offspring: list[Solution] = []

    def evaluated(self) -> bool:
        return self.fitness > float("-inf")

    def parents(self) -> Sequence[Solution]:
        return self._parents

    def offspring(self) -> Sequence[Solution]:
        return self._offspring

    def add_parent(self, parent: Solution) -> None:
        self._parents.append(parent)

    def add_offspring(self, offspring: Solution) -> None:
        self._offspring.append(offspring)

    def __lt__(self, other: Solution) -> bool:
        return self.fitness < other.fitness

    def __le__(self, other: Solution) -> bool:
        return self.fitness <= other.fitness

    def __str__(self) -> str:
        return f"value={self.value}"

    def __repr__(self) -> str:
        return f"DummySolution({self})"


class DummyFeature(Feature):
    def __init__(
        self, name: str, min_value: float = 0.0, max_value: float = 1.0
    ) -> None:
        self.name = name
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self) -> str:
        return f"{self.name}(min={self.min_value},max={self.max_value})"

    def __repr__(self) -> str:
        return f"Feature: {self}"

    def value_for(self, solution: Solution) -> float:
        if self.name == "feature1":
            return solution.fitness * 0.1
        else:
            return solution.fitness * 0.2


class DummySolutionSpace(SolutionSpace):
    def __init__(self) -> None:
        self._solutions = []

    def empty(self) -> bool:
        return len(self._solutions) == 0

    def add(self, solution: Solution) -> None:
        self._solutions.append(solution)

    def solutions(self) -> Sequence[Solution]:
        return self._solutions

    def best_solution(self) -> Solution:
        if not self._solutions:
            raise IndexError("No solutions in space")

        return max(self._solutions, key=lambda s: s.fitness)

    def select(self, k: int = 1) -> Sequence[Solution]:
        if k <= 0:
            return []

        return self._solutions[:k]

    def clear(self) -> None:
        self._solutions.clear()


class DummyGenerator(Generator):
    def __init__(self) -> None:
        self.generation: int
        self._rnd: random.Random
        self.reset()

    def reset(self) -> None:
        """Reset the generator's state."""
        self.generation = 0
        self._rnd = random.Random()

    def generate(self, k: int = 1) -> Sequence[DummySolution]:
        """Generate n random integer solutions."""
        solutions: list[DummySolution] = []

        for _ in range(k):
            value = self._rnd.randint(-10, 10)
            solution = DummySolution(value=value)
            solutions.append(solution)

        self.generation += 1

        return solutions

    def select_and_recombine(
        self, solution_space: SolutionSpace
    ) -> list[DummySolution]:
        """Select solutions and recombine them to create new solutions."""
        # For this simple problem, just select one solution and modify it
        if solution_space.empty():
            return []

        # Select a random solution
        solution = self._rnd.choice(solution_space.solutions())

        # Recombine: randomly add +1 or -1
        if self._rnd.random() > 0.2:
            new_value = np.sign(solution.value) * (abs(solution.value) - 1)
        else:
            new_value = np.sign(solution.value) * (abs(solution.value) + 1)

        new_value = max(-10, min(10, new_value))
        new_solution = DummySolution(value=int(new_value))

        return [new_solution]


class DummyEvaluator(Evaluator):
    def evaluate(self, solution: DummySolution) -> float:
        """Evaluate a solution. Closer to zero is better."""
        return float(-abs(solution.value))

    def reset(self) -> None:
        pass


class DummyOptimizer(Optimizer):
    def __init__(self) -> None:
        self.best_solution: DummySolution | None = None

    def optimize(self, n_generations: int, target_fitness: float) -> Solution:
        # Simple optimizer that just returns a solution with target fitness
        self.best_solution = DummySolution(fitness=target_fitness)

        return self.best_solution

    def reset(self) -> None:
        self.best_solution = None


class DummySolutionLogger(SolutionLogger):
    def __init__(self) -> None:
        self.log: list[str] = []

    def solution(self, solution: Solution) -> None:
        self.log.append(
            f"Solution: fitness={solution.fitness}, generation={solution.generation}"
        )

    def feature(self, solution: Solution, feature_name: str, value: float) -> None:
        self.log.append(f"Feature {feature_name} for solution: {value}")

    def evaluation(self, solution: Solution) -> None:
        self.log.append(f"Evaluation: fitness={solution.fitness}")


class DummyPromptTemplate(LanguageModelPromptTemplate):
    def __init__(self, template_str: str) -> None:
        self.template_str = template_str

    @classmethod
    def load(cls, path: str | Path) -> "DummyPromptTemplate":
        return cls("{{ prompt }} is followed by {{ response }}")

    def apply(self, values: dict[str, Any]) -> str:
        result = self.template_str

        for key, value in values.items():
            result = result.replace(f"{{{{ {key} }}}}", str(value))

        return result


class DummyLanguageModelResponse(LanguageModelResponse):
    def __init__(
        self,
        conversation_id: str,
        round_num: int,
        model_id: str,
        role: str = "default",
        prompt_text: str = "test_prompt",
        response_text: str = "test_response",
    ) -> None:
        self.conversation_id = conversation_id
        self.round = round_num
        self.model = model_id
        self.role = role
        self._prompt = prompt_text
        self._response = response_text
        self._reasoning = ""

    def prompt(self) -> str:
        return self._prompt

    def text(self) -> str:
        return self._response

    def reasoning(self) -> str:
        return self._reasoning

    def system_prompt(self) -> str:
        return "System prompt"


class DummyLanguageModel(LanguageModel):
    def __init__(
        self, model_id: str, role: str = "default", system_prompt: str | None = None
    ) -> None:
        self.model_id = model_id
        self.role = role
        self.conversation_history: list[tuple[str, str]] = []
        self.system_prompt = system_prompt

    def options(self) -> dict[str, float | int]:
        return {"temperature": 0.7, "top_k": 20}

    def conversation(self, system: str | None = None) -> "DummyLanguageModel":
        new_model = DummyLanguageModel(self.model_id, self.role, system_prompt=system)

        return new_model

    def prompt(
        self,
        prompt: str,
        system: str | None = None,
        options: dict[str, float | int] | None = None,
    ) -> "DummyLanguageModelResponse":
        response = DummyLanguageModelResponse(
            conversation_id="test_conv",
            round_num=len(self.conversation_history) + 1,
            model_id=self.model_id,
            role=self.role,
            prompt_text=prompt,
        )
        self.conversation_history.append((prompt, response.text()))

        return response

    def clear(self) -> None:
        self.conversation_history.clear()


class DummyLanguageModelLogger(LanguageModelLogger):
    def __init__(self) -> None:
        self.log: list[str] = []

    def response(self, response: DummyLanguageModelResponse) -> None:
        self.log.append(f"Response from {response.model}: {response.text()}")


class DummyEmbeddingLanguageModel(EmbeddingLanguageModel):
    def embed(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])
