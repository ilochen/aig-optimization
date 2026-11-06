"""Current entry point. Original course interfaces are archived under legacy/."""
from aigopt.abc import ABC, Action, actions
from aigopt.evaluator import Evaluator
from aigopt.search import search

if __name__ == "__main__":
    from aigopt.__main__ import main
    main()
