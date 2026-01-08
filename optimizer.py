from collections.abc import Sequence
import heapq
import logging
import numpy as np
import random

from protocols import (
    Feature,
    Evaluator,
    GenerationException,
    Generator,
    Solution,
    SolutionLogger,
)

logger = logging.getLogger(__name__)

MINUS_INF = float("-inf")


class SearchGrid:
    """Discretized search feature space.

    Given a list of features f_1, ..., f_k, an instance of this class
    conceptually builds a (potentially unbounded) discretized feature space,
    where the granularity is chosen by the user for each feature. Each discrete
    cell of such a space is represented by a "centroid" point. For example, this
    is a 4x3 grid:

          2 (3.,10.)  (4.,10.)  (5.,10.)  (6.,10.)
          1 (3., 6.)  (4., 6.)  (5., 6.)  (6., 6.)
          0 (3., 2.)  (4., 2.)  (5., 2.)  (6., 2.)
               0         1         2         3

    In this example, the feature on the first dimension, whose values range from
    3 to 6, has been discretized with a step equal to 1.0, while the feature on
    the second dimension, whose values vary between 2 and 10, has been
    discretized with a coarser step equal to 4.0.

    Note that each dimension can have its own discretization step, but the
    discretization must be uniform along a given dimension.

    The grid may be bounded or unbounded. Negative indexes are allowed by
    default, but coordinates may be clipped to the range defined by the
    features. With reference to the example above, the grid point at coordinates
    (-1,-2) would be clipped to (0,0) (corresponding to feature vector (3., 2.))
    and the point (1,5) would be clipped to (1,2) (corresponding to feature
    vector (4., 10.)).

    The main operation that this data structure enables is "snapping" an
    arbitrary feature to the grid, that is, returning the coordinates of a point
    on the grid that is nearest to an input feature vector. This operation is
    used, for example, by the MAP-Elites algorithm to assign a solution to
    a particular region of the search space during optimization.

    Attributes
    ----------
    features: list[Feature]
        The list of features labeling the dimensions of the search grid.
    shape: numpy.array
        The number of discrete points along each dimension of the search grid.
        The size of this array is  equal to the number of features.
    min: numpy.array
        The minimum values for each dimension. The size of this array coincides
        with the number of features.
    max: numpy.array
        The maximum values for each dimension. The size of this array coincides
        with the number of features.
    step: numpy.array
        The grid step in each dimension. The size of this array coincides with
        the number of features.

    Parameters
    ----------
    features: list[Feature]
        A list of features of interest. For example, these could be length and
        entropy of a string. The bounds of each dimension of the search grid are
        determined by the valid ranges of these features.
    shape: tuple[int, ...] | int
        The number of discrete points along each dimension. If this is a tuple,
        the size of the tuple must be equal to the number of features. If this
        is a single integer, the same number of points is created along each
        dimension.

        NOTE: The search grid must have at least two points in each dimension.
    """

    def __init__(self, features: list[Feature], shape: int | tuple[int, ...]) -> None:
        self.features = features
        self.shape: np.ndarray

        if isinstance(shape, int):
            self.shape = np.array([shape] * len(features))
        else:
            self.shape = np.array(shape)

        assert len(self.features) == len(self.shape)
        assert np.all(self.shape > 1)  # At least two points per dimension

        self.min = np.array([feat.min_value for feat in self.features])
        self.max = np.array([feat.max_value for feat in self.features])
        self.step = (self.max - self.min) / (self.shape - 1)

    def at(self, cell: tuple[int, ...], clip: bool = False) -> np.ndarray:
        """Return the feature vector at the given cell coordinates.

        For example, given the following 4x3 grid:

          2  (3.,10.)  (4.,10.)  (5.,10.)  (6.,10.)
          1  (3., 6.)  (4., 6.)  (5., 6.)  (6., 6.)
          0  (3., 2.)  (4., 2.)  (5., 2.)  (6., 2.)
                0         1         2         3

        at((0,0)) returns (3., 2.), and at((2,1)) returns (5., 6.).

        Parameters
        ----------
        cell: tuple[int, ...]
            A tuple of indexes on the search grid.
        clip bool, default is False
            Whether the returned value should be within the range defined by the
            features.

            With reference to the example above, requesting the point at
            position (4,3), which is outside the finite grid, by default returns
            the feature vector (7., 14.). But if `clip` is `True` then the
            method returns (6., 10.) instead, that is, the nearest point of the
            finite grid.

            Similarly, negative indexes are allowed by default. Requesting the
            grid point at (-1,-1) returns the feature vector (2., -2.) by
            default. If `clip` is set to `True` then (3., 2.) is returned
            instead.

        Returns
        -------
        numpy.array
            A 1-dimensional array of size equal to the number of features of the
            grid.
        """
        grid_point = self.min + cell * self.step

        if clip:
            np.clip(grid_point, a_min=self.min, a_max=self.max, out=grid_point)

        return grid_point

    def centroids(self, dim: int) -> np.ndarray:
        """Return the grid values along the given dimension.

        NOTE: calling this method takes time proportional to the number of
        points along the requested dimension, hence it may be slow!

        Parameters
        ----------
        dim: int
            The dimension along which the coordinates should be returned. The
            returned values are limited by the valid ranges of the correponding
            features.

        Returns
        -------
        numpy.array
            A 1-dimensional array of equidistant values.
        """
        return np.linspace(
            start=self.min[dim],
            stop=self.max[dim],
            num=self.shape[dim],
            endpoint=True,
        )

    def snap(self, feature: np.ndarray, clip: bool = False) -> tuple[int, ...]:
        """Return the indexes of the grid point nearest to a given feature vector.

        Consider, for example, the following 4x3 grid:

          2 (3.0,10.0)  (4.0,10.0)  (5.0,10.0)  (6.0,10.0)
          1 (3.0, 6.0)  (4.0, 6.0)  (5.0, 6.0)  (6.0, 6.0)
          0 (3.0, 2.0)  (4.0, 2.0)  (5.0, 2.0)  (6.0, 2.0)
                0           1           2           3

        The following are some examples of outputs for given inputs:

            feature         returned index vector   nearest grid point
            (3.1, 2.4)  =>  (0,0)                   (3.0, 2.0)
            (5.6,11.0)  =>  (3,2)                   (6.0,10.0)
            (0.0, 6.5)  =>  (0,1)                   (3.0, 6.0)

        Parameters
        ----------
        feature: np.array
            An 1-dimensional array of real values. The array size must be equal
            to the number of dimensions of the grid.
        clip: bool, default is False
            Whether the returned value should remain within the range defined by
            the features. See `SearchGrid.at()` for details.

        Returns
        -------
        tuple[int, ...]
            The indexes of the point in the search grid that is nearest to the
            input feature.
        """
        assert feature.shape == (len(self.features),)

        # "Stepify" the feature, that is, find, for each dimension, the index of
        # the largest grid value less than the corresponding feature component.
        index = ((feature - self.min) // self.step).astype(int)

        # Calculate the corresponding grid value
        value = self.min + index * self.step

        # For each feature component, determine whether it is closer to the grid
        # value "on its left" or to the grid value "on its right", and update
        # the index so that it always points to the nearest value.
        #
        # For example, in this 3x3 grid, point X is initially "snapped" to the
        # grid point at x=1, y=1. But the point is actually closer to x=2, y=1,
        # so the first index must be updated.
        #
        #      2  +       +        +
        #
        #                       X
        #      1  +       +        +
        #
        #
        #      0  +       +        +
        #
        #         0       1        2

        diff_l = np.abs(feature - value)
        diff_r = np.abs(feature - value - self.step)
        index = np.where(diff_l > diff_r, index + 1, index)

        if clip:  # Clip the indexes to the allowed range
            index = np.clip(index, a_min=0, a_max=self.shape - 1)

        return tuple(map(int, index))


class Elite:
    """An elite is a locally optimal solution in a solution space.

    Parameters
    ----------
    solution: Solution
        The elite solution.
    cell: tuple[int, ...]
        The coordinates of the elite in the solution space it belongs to. The
        coordinates are indexes in the search grid, one index for each dimension
        (that is, feature) of the grid.
    """

    def __init__(self, solution: Solution, cell: tuple[int, ...]) -> None:
        self._solution = solution
        self._cell = cell

    @property
    def solution(self) -> Solution:
        return self._solution

    @property
    def cell(self) -> tuple[int, ...]:
        return self._cell

    @property
    def fitness(self) -> float:
        return self._solution.fitness

    def __lt__(self, other: "Elite") -> bool:
        return self._solution.fitness < other._solution.fitness

    def __le__(self, other: "Elite") -> bool:
        return self._solution.fitness <= other._solution.fitness

    def __str__(self) -> str:
        return f"{self._solution} (fitness={self._solution.fitness})"

    def __repr__(self) -> str:
        return f"Elite: {self}"


class SolutionGrid:  # Conforms to SolutionSpace[Solution]
    """A map of solutions in a discrete solution space.

    Store the solutions found by an optimizer and their location in
    a discretized search space. Also keep track of the current globally best
    solution.

    Attributes
    ----------
    elite: list[Elite]
        The elite solutions.
    features: list[Feature]
        The list of features labeling the dimensions of the search grid.
    search_grid: SearchGrid
        The underlying discretized search space. Read-only.

    Parameters
    ----------
    features: list[Feature]
        A list of features of interest.
    shape: tuple[int, ...]
        The number of point in each dimension of the feature space. The higher
        the number of points, the finer the granularity of the discretization of
        the search space. The grid shape must have as many elements as the
        number of features. Note that the search space is bounded by the valid
        ranges associated to each feature.
    logger: SolutionLogger | None, default is None
        A logger for solutions, extracted features and evaluations.
    """

    def __init__(
        self,
        features: list[Feature],
        shape: tuple[int, ...],
        logger: SolutionLogger | None = None,
    ) -> None:
        self._search_grid = SearchGrid(features=features, shape=shape)
        self._log = logger

        self._solutions: list[Solution]  # Keeps every solution ever added
        self._elite_map: dict[tuple[int, ...], Elite]  # Sparse map of elites
        self._elite_heap: list[Elite]  # Maintains elites as a max-heap

        self.clear()  # Initialize the above members

    @property
    def elite(self) -> list[Elite]:
        return self._elite_heap

    @property
    def features(self) -> list[Feature]:
        return self._search_grid.features

    @property
    def search_grid(self) -> SearchGrid:
        """The underlying search grid."""
        return self._search_grid

    def empty(self) -> bool:
        """Check whether the solution space is empty.

        Returns
        -------
        bool
            `True` if the solution space is empty; `False` otherwise.
        """
        return len(self._elite_heap) == 0

    def select(self, k: int = 1) -> list[Solution]:
        """Return random elites without replacement."""
        # TODO: favor higher fitness candidates (take softmax of fitness or smth like that)
        return [
            e.solution
            for e in random.sample(self._elite_heap, k=min(k, len(self._elite_heap)))
        ]

    def feature_descriptor(self, solution: Solution) -> np.ndarray:
        return np.array([f.value_for(solution) for f in self.features])

    def best_solution(self) -> Solution:
        """The globally best solution.

        Returns
        -------
        Solution
            The overall best elite. If no solutions exist yet, return `None`.

        Raises
        ------
        IndexError
            If no solutions have been added yet. Check whether the solution
            space is empty before calling this method.
        """
        return self._elite_heap[0].solution

    def top_k(self, k: int) -> list[Elite]:
        """Return the k best elites.

        If you need the globally best solution, calling
        `SolutionGrid.best_solution()` is more efficient than using this
        method.

        Note that this method returns the top **elites**, that is, the best
        among the dominant solutions—not the best among all solutions ever
        added. For example, if N>1 solutions have been added to the solution
        space, but they are all mapped to the same cell of the solution space,
        this method will return one solution (the only elite) regardless of how
        many solutions are requested.

        Parameters
        ----------
        k: int
            The number of elites to return. Note that if the solution space
            contains less than k elites, the number of returned solutions will
            be less than k. An empty list is returned if k is less than or equal
            to zero.

        Returns
        -------
        list[Elite]
            A list of at most k best elites.
        """
        # TODO: check whether copying the heap and repeatedly calling
        # heappop_max() is faster than this. Or maybe popping and pushing again
        # is the best way.
        return heapq.nlargest(k, self._elite_heap)

    def solutions(self) -> list[Solution]:
        """Return all the solutions in the order in which they were added."""
        return self._solutions

    def add(self, solution: Solution) -> None:
        """Update the solution space with a new solution.

        Add a solution and update the map of elites with a new solution if it is
        a local optimum. The fitness of the solution should have already been
        computed.

        Find the appropriate cell in the discrete space for the given solution.
        If that cell is currently empty or if the associated solution is worse
        than the new solution, update the cell by replacing the old solution
        with the new one. Otherwise, if the input solution is not better than
        a currently existing solution, leave the map of elites unmodified.

        Besides, if the given solution's fitness improves over the current
        globally best fitness, make the given solution the new globally best
        solution.

        Parameters
        ----------
        solution: Solution
            The solution to add.
        """
        self._solutions.append(solution)
        feature_descriptor = self.feature_descriptor(solution)

        if self._log:
            self._log.solution(solution)
            self._log.evaluation(solution)

            for i, feature in enumerate(self._search_grid.features):
                self._log.feature(
                    solution=solution,
                    feature_name=feature.name,
                    value=feature_descriptor[i],
                )

        # Find the location of the appropriate cell in the search space
        cell: tuple[int, ...] = self._search_grid.snap(feature_descriptor, clip=True)

        # If the appropriate cell is empty or its occupant's performance is less
        # than the fitness of the new solution then store the new solution in
        # the map of elites at the location determined by its feature
        # descriptor, and discard the old solution. Also update the heap so that
        # the best solutions stay on top.
        if cell in self._elite_map:
            old_elite = self._elite_map[cell]

            # TODO: might be >= to prefer newer solutions
            if solution.fitness > old_elite.fitness:
                # Remove the old elite from the heap
                self._elite_heap.remove(old_elite)
                # Rebuild the heap
                heapq.heapify_max(self._elite_heap)
                # Push the new elite to the heap
                new_elite = Elite(solution, cell)
                heapq.heappush_max(self._elite_heap, new_elite)
                # Finally, update the elite map
                self._elite_map[cell] = new_elite
        else:
            new_elite = Elite(solution, cell)
            self._elite_map[cell] = new_elite
            heapq.heappush_max(self._elite_heap, new_elite)

        if solution is self.best_solution():
            logger.debug(
                f"DEBUG [optimizer]: CURRENT BEST is {solution} (fitness={solution.fitness})"
            )

    def clear(self) -> None:
        """Clear all the solutions."""
        # A sparse grid map from locations in the search space to the best
        # solutions found in the corresponding cells.
        self._solutions: list[Solution] = []
        self._elite_map: dict[tuple[int, ...], Elite] = {}
        self._elite_heap: list[Elite] = []


class MAPElitesOptimizer:  # Conforms to Optimizer
    """Automatically explore a feature space for high-performing solutions.

    Note that an optimizer maintains its own state of solutions. If you want to
    run multiple optimizations concurrently, you should create multiple
    instances of this class rather than creating a single instance and passing
    it around to multiple optimization tasks.

    Reference:
        J-B Mouret and J Clune, Illuminating search spaces by mapping elites, 2015.


    Attributes
    ----------
    elite: dict[tuple[int, ...], Solution]
        The map of elites.
    features: list[Feature]
        The features of interest in a solution.
    solution_grid: SolutionGrid
        The solution grid. Read-only.
    n_steps: int
        Total number of iterations carried out by the optimizer.

    Parameters
    ----------
    features: list[Feature]
        A list of features of interest.
    grid_shape: tuple[int, ...]
        The number of point in each dimension of the feature space. The higher
        the number of points, the finer the granularity of the discretization of
        the search space. The grid shape must have as many elements as the
        number of features. Note that the search space is bounded by the valid
        ranges associated to each feature.
    generator: Generator
        A solution generator. This object must be able to generate solutions for
        which the given features can be extracted. For example, if the generator
        generates textual solutions, but the optimizer is told that one of the
        features is the numerical mean, that is an inconsistency that leads to
        a runtime error.
    evaluator: Evaluator
        A solution evaluator.
    n_initial_solutions: int, default is 1
        Number of initial solutions to generate. These are randomly generated.
        Note that the solutions added via `MapElitesOptimizer.add_solutions()`
        count as initial solutions, too. For example, if `n_initial_solutions`
        is 1 and `add_solutions()` is used to add a solution then, when
        `optimize()` is called, no new initial solutions will be generated
        because the number of initial solutions has already been reached.
    logger: SolutionLogger | None, default is None
        A logger for solutions, extracted features and evaluations.
    """

    def __init__(
        self,
        features: list[Feature],
        grid_shape: tuple[int, ...],
        generator: Generator,
        evaluator: Evaluator,
        n_initial_solutions: int = 1,
        logger: SolutionLogger | None = None,
    ) -> None:
        assert len(features) == len(grid_shape)

        self._solution_grid = SolutionGrid(
            features=features, shape=grid_shape, logger=logger
        )
        self._generator = generator
        self._evaluator = evaluator
        self._n_initial_solutions = n_initial_solutions

    @property
    def elite(self) -> list[Solution]:
        return [e.solution for e in self._solution_grid.elite]

    @property
    def solution_grid(self) -> SolutionGrid:
        return self._solution_grid

    @property
    def features(self) -> list[Feature]:
        return self._solution_grid.features

    def reset(self) -> None:
        """Reset the optimizer to its initial state.

        Call this method if you want to discard any previously found solution
        and start an optimization process from scratch. This also resets the
        generator's state, and evaluator's state, and the step counter.
        """
        self._solution_grid.clear()
        self._generator.reset()
        self._evaluator.reset()

    def add_solutions(self, solutions: Sequence[Solution]) -> None:
        """Add solutions to the solution space of the optimizer.

        This method can be used to populate the optimizer with an initial set of
        solutions. If the solutions have not been evaluated yet then this method
        performs their evaluation as part of the process for adding them to the
        solution space.
        """
        for solution in solutions:
            if not solution.evaluated():
                solution.fitness = self._evaluator.evaluate(solution)

            self._solution_grid.add(solution)

    def top_k(self, k: int) -> list[Solution]:
        """Return the k best solutions."""
        return [e.solution for e in self._solution_grid.top_k(k)]

    def optimize(
        self, n_generations: int, target_fitness: float = float("inf")
    ) -> Solution:
        """Run the optimizer for a given number of generations.

        Evolve up to `n_generations` new generations of solutions. Each
        generation may generate one or more solutions. At least
        `MAPElitesOptimizer.n_initial_solutions` solutions are guaranteed to
        exist after the first generation. Note, however, that the number of
        elites (locally optimal solutions) may be less than that.

        If a solution with the given target fitness score or higher is found
        before `n_generations` iterations, return early.

        This method may be called multiple times consecutively, which allows you
        to continue optimizing. For example, you may run the optimization for
        N generations, inspect the result and if you are not satisfied you may
        call this method again to keep optimizing from where you left.

        To reset the optimization process, call `MAPElitesOptimizer.reset()`.

        Parameters
        ----------
        n_generations: int
            Number of generations. This must be > 0.
        target_fitness: float, default is float("inf")
            If this value is provided, then the optimization process may stop
            early if a solution is found whose fitness equals or exceeds the
            target fitness. The default is to stop after `n_generations`
            iterations have been executed.

        Returns
        -------
        Solution
            The best solution that could be found. To obtain all the (locally)
            best solutions that were found, you may access the
            `MAPElitesOptimizer.solution_grid` attribute after the optimization.

        Raises
        ------
        GenerationException
            If an error occurs while trying to generate or evolve solutions.
        EvaluationException
            If an error occurs while trying to evaluate a solution.
        """
        # Total number of solutions that have already been generated
        n_solutions = len(self._solution_grid.solutions())
        n_gen = 0

        # Generate a sufficient number of initial solutions
        if n_solutions < self._n_initial_solutions:
            new_solutions = self._generator.generate(
                self._n_initial_solutions - n_solutions
            )
            self.add_solutions(new_solutions)
            n_gen += 1

        while n_gen < n_generations:
            new_solutions = self._generator.select_and_recombine(
                solution_space=self._solution_grid
            )
            self.add_solutions(new_solutions)

            n_gen += 1

            if self._solution_grid.empty():
                continue  # For some reason, no solutions were generated

            best_solution = self._solution_grid.best_solution()

            if best_solution.fitness >= target_fitness:
                break

        if n_gen < n_generations:
            logger.debug(f"MAP-Elites: Early stop after {n_gen} generations.")

        if self._solution_grid.empty():
            raise GenerationException(f"No solutions after {n_generations} steps.")

        return self._solution_grid.best_solution()


# vim: tw=80
