from time import time
from typing import Type, TYPE_CHECKING, Optional, List, Sequence

from deepproblog.arithmetic_circuit import ArithmeticCircuit
from deepproblog.engines import Engine
from deepproblog.query import Query
from deepproblog.semiring import Result
from deepproblog.semiring.graph_semiring import GraphSemiring, Semiring
from deepproblog.utils.cache import Cache
from problog.formula import LogicFormula
from problog.logic import Term

import globals

if TYPE_CHECKING:
    from deepproblog.model import Model

class SolverException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Solver(object):
    """
    A class that bundles the different steps of inference.
    """

    def __init__(
        self,
        model: "Model",
        engine: Engine,
        semiring: Type[Semiring] = GraphSemiring,
        cache: bool = False,
        cache_root: Optional[str] = None,
    ):
        """
        :param model: The model in which queries will be evaluated.
        :param engine: The engine that will be used to ground queries.
        :param semiring: The semiring that will be used to evaluate the arithmetic circuits.
        :param cache: If true, then arithmetic circuits will be cached.
        :param cache_root: If cache_root is not None, then the cache is persistent and is saved to that directory.
        """
        if cache and not engine.can_cache():
            raise SolverException(
                "Caching is enabled, but {} cannot cache.".format(type(engine))
            )

        self.cache = Cache[Query, ArithmeticCircuit](
            func=self.build_ac,
            cache=cache,
            cache_root=cache_root,
            key_func=lambda x: x.query,
        )
        self.engine = engine
        self.model = model
        self.program = self.engine.prepare(model.program)
        self.semiring = semiring

    def build_ac(self, q: Query) -> ArithmeticCircuit:
        """
        Builds the arithmetic circuit.
        :param q: The query for which to build the arithmetic circuit.
        :return: The arithmetic circuit for the given query.
        """
        start = time()

        ground = self.engine.ground(q, label=LogicFormula.LABEL_QUERY)
        ground_time = time() - start
        ac = ArithmeticCircuit(ground, self.semiring, ground_time=ground_time)
        return ac

    def solve(self, batch: Sequence[Query]) -> List[Result]:
        """
        Performs inference for a batch of queries.
        :param batch: A list of queries to perform inference on.
        :return: A list of results for the given queries.
        """
        self.engine.tensor_store.clear()
        # Build ACs
        acs = []
        for q in batch:
            query_in_mapping = (str(q) in globals.query_story_mapping)
            globals.record_networks = query_in_mapping
            if query_in_mapping:
                with open("rel_extract_calibration_validation.txt", "a") as f:
                    f.write(f"{globals.query_story_mapping[str(q)].get_relations()}\n")
                    f.write(f"{globals.query_story_mapping[str(q)].get_genders()}\n")
                with open("gender_net_calibration_validation.txt", "a") as g:
                    g.write(f"{globals.query_story_mapping[str(q)].get_genders()}\n")
            acs.append(
                self.cache.get(q)
            )
            globals.record_networks = False

        # Evaluate ACs. Evaluate networks if necessary
        result = []
        for i, ac in enumerate(acs):
            result.append(
                ac.evaluate(self.model, batch[i].substitution)
            )
        semirings = [r.semiring for r in result]
        self.engine.perform_count(batch, (acs, semirings))
        return result

    def get_tensor(self, term: Term):
        return self.engine.get_tensor(term)

    def get_hyperparameters(self) -> dict:
        parameters = dict()
        parameters["engine"] = self.engine.get_hyperparameters()
        parameters["semiring"] = self.semiring.__name__
        return parameters
