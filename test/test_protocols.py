import unittest

import numpy as np

from protocols import (
    EmbeddingLanguageModel,
    Evaluator,
    Feature,
    Generator,
    LanguageModel,
    LanguageModelLogger,
    LanguageModelPromptTemplate,
    LanguageModelResponse,
    Optimizer,
    Solution,
    SolutionLogger,
    SolutionSpace,
)

from test.fixtures import (
    DummyEmbeddingLanguageModel,
    DummyEvaluator,
    DummyFeature,
    DummyGenerator,
    DummyLanguageModel,
    DummyLanguageModelLogger,
    DummyLanguageModelResponse,
    DummyOptimizer,
    DummyPromptTemplate,
    DummySolution,
    DummySolutionLogger,
    DummySolutionSpace,
)


class TestLanguageModelProtocols(unittest.TestCase):
    def test_prompt_template_protocol(self) -> None:
        """Test that DummyPromptTemplate conforms to LanguageModelPromptTemplate protocol."""
        loaded = DummyPromptTemplate.load("test_path")
        self.assertIsInstance(loaded, LanguageModelPromptTemplate)

        template = DummyPromptTemplate("Hello {{ word }}")
        self.assertIsInstance(template, LanguageModelPromptTemplate)

        result = template.apply({"word": "world!"})
        self.assertEqual(result, "Hello world!")

    def test_language_model_response_protocol(self) -> None:
        """Test that DummyLanguageModelResponse conforms to LanguageModelResponse protocol."""
        response = DummyLanguageModelResponse("conv1", 1, "modelA")
        self.assertIsInstance(response, LanguageModelResponse)

        # Test attributes
        self.assertEqual(response.conversation_id, "conv1")
        self.assertEqual(response.round, 1)
        self.assertEqual(response.model, "modelA")

        # Test methods
        self.assertEqual(response.prompt(), "test_prompt")
        self.assertEqual(response.text(), "test_response")
        self.assertEqual(response.reasoning(), "")

    def test_language_model_protocol(self) -> None:
        """Test that DummyLanguageModel conforms to LanguageModel protocol."""
        model = DummyLanguageModel("test_model")
        self.assertIsInstance(model, LanguageModel)

        # Test attributes
        self.assertEqual(model.model_id, "test_model")

        # Test options
        options = model.options()
        self.assertEqual(options["temperature"], 0.7)

        # Test conversation
        new_conv = model.conversation()
        self.assertIsInstance(new_conv, LanguageModel)

        # Test prompt
        response = model.prompt("Test prompt text")
        self.assertIsInstance(response, LanguageModelResponse)

        # Test clear
        model.clear()
        self.assertEqual(len(model.conversation_history), 0)

    def test_language_model_logger_protocol(self) -> None:
        """Test that DummyLanguageModelLogger conforms to LanguageModelLogger protocol."""
        logger = DummyLanguageModelLogger()
        self.assertIsInstance(logger, LanguageModelLogger)

        response = DummyLanguageModelResponse("conv1", 1, "modelA")
        logger.response(response)

        self.assertEqual(len(logger.log), 1)

    def test_embedding_language_model_protocol(self) -> None:
        """Test that DummyEmbeddingLanguageModel conforms to EmbeddingLanguageModel protocol."""
        model = DummyEmbeddingLanguageModel()
        self.assertIsInstance(model, EmbeddingLanguageModel)

        # Test embed
        embedding = model.embed("test text")
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 3)


class TestProtocols(unittest.TestCase):
    def test_solution_protocol(self) -> None:
        """Test that DummySolution conforms to Solution protocol."""
        s = DummySolution(fitness=5.0, generation=1)
        self.assertIsInstance(s, Solution)

        # Test attributes
        self.assertEqual(s.fitness, 5.0)
        self.assertEqual(s.generation, 1)

        # Test methods
        self.assertTrue(s.evaluated())
        self.assertEqual(s.parents(), [])
        self.assertEqual(s.offspring(), [])

        # Test comparison
        s2 = DummySolution(fitness=3.0, generation=1)
        self.assertTrue(s != s2)
        self.assertFalse(s == s2)
        self.assertTrue(s > s2)
        self.assertTrue(s >= s2)
        self.assertFalse(s < s2)
        self.assertFalse(s <= s2)

    def test_feature_protocol(self) -> None:
        """Test that DummyFeature conforms to Feature protocol."""
        feature = DummyFeature("test_feature", 0.0, 10.0)
        self.assertIsInstance(feature, Feature)

        # Test attributes
        self.assertEqual(feature.name, "test_feature")
        self.assertEqual(feature.min_value, 0.0)
        self.assertEqual(feature.max_value, 10.0)

        # Test methods
        self.assertEqual(str(feature), "test_feature(min=0.0,max=10.0)")
        self.assertEqual(repr(feature), "Feature: test_feature(min=0.0,max=10.0)")

        # Test value_for
        solution = DummySolution(fitness=2.0)
        self.assertEqual(feature.value_for(solution), 0.4)

    def test_solution_space_protocol(self) -> None:
        """Test that DummySolutionSpace conforms to SolutionSpace protocol."""
        space = DummySolutionSpace()
        self.assertIsInstance(space, SolutionSpace)

        # Test empty space
        self.assertTrue(space.empty())
        self.assertEqual(space.solutions(), [])

        # Add solutions
        s1 = DummySolution(fitness=5.0)
        s2 = DummySolution(fitness=10.0)
        space.add(s1)
        space.add(s2)

        self.assertFalse(space.empty())
        self.assertEqual(len(space.solutions()), 2)

        # Test best_solution
        self.assertEqual(space.best_solution(), s2)

        # Test select
        selected = space.select(k=1)
        self.assertEqual(len(selected), 1)

        # Test clear
        space.clear()
        self.assertTrue(space.empty())

    def test_generator_protocol(self) -> None:
        """Test that DummyGenerator conforms to Generator protocol."""
        generator = DummyGenerator()
        self.assertIsInstance(generator, Generator)

        # Test generate
        solutions = generator.generate(k=2)

        self.assertEqual(len(solutions), 2)

        space = DummySolutionSpace()

        for s in solutions:
            self.assertIsInstance(s, DummySolution)
            space.add(s)

        # Test select_and_recombine
        new_solutions = generator.select_and_recombine(space)

        self.assertEqual(len(new_solutions), 1)

        # Test reset
        generator.reset()

        self.assertEqual(generator.generation, 0)

    def test_evaluator_protocol(self) -> None:
        """Test that DummyEvaluator conforms to Evaluator protocol."""
        evaluator = DummyEvaluator()
        self.assertIsInstance(evaluator, Evaluator)

        # Test evaluate
        solution = DummySolution(value=42, generation=2)
        fitness = evaluator.evaluate(solution)

        self.assertEqual(fitness, -42.0)

        # Test reset
        evaluator.reset()

    def test_optimizer_protocol(self) -> None:
        """Test that DummyOptimizer conforms to Optimizer protocol."""
        optimizer = DummyOptimizer()
        self.assertIsInstance(optimizer, Optimizer)

        # Test optimize
        solution = optimizer.optimize(n_generations=10, target_fitness=95.0)
        self.assertIsInstance(solution, Solution)
        self.assertEqual(solution.fitness, 95.0)

        # Test reset
        optimizer.reset()
        self.assertIsNone(optimizer.best_solution)

    def test_solution_logger_protocol(self) -> None:
        """Test that DummySolutionLogger conforms to SolutionLogger protocol."""
        logger = DummySolutionLogger()
        self.assertIsInstance(logger, SolutionLogger)

        # Test logging methods
        solution = DummySolution(fitness=8.0)
        logger.solution(solution)
        logger.feature(solution, "test_feature", 42.0)
        logger.evaluation(solution)

        self.assertEqual(len(logger.log), 3)


if __name__ == "__main__":
    unittest.main()
