from deepproblog.engines import ApproximateEngine
from deepproblog.model import Model
from deepproblog.heuristics import geometric_mean
from problog.logic import Term, Constant, Var
from deepproblog.query import Query

def _create_model(program, cache = False) -> Model:
    """Setup code: Load a program minimally"""
    model = Model(program, [], load=False)
    engine = ApproximateEngine(model, 10, geometric_mean)
    model.set_engine(engine, cache = cache)
    return model

def test_assignment():
    program = """
a(X) :- X=3.
    """
    def evaluate(x):
        print("Inside evaluate", x)
        return Constant(x ** 2)

    model = _create_model(program)
    model.register_foreign(evaluate, "evaluate", 1, 1)
    q = Query(Term("a", Var("X")))
    i = model.solve([q])
    r = model.solve([q])[0].result[Term("a", Constant(3))]
    breakpoint()

if __name__ == "__main__":
  test_assignment()