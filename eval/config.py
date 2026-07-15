"""
Retriever config getter for eval — re-exports the dataclass so run_eval.py
can construct one without circular imports through parse_args().
"""
# This file exists only so eval/run_eval.py can do:
#   from config import RetrieverConfig
# without hitting the argparse machinery. The actual config is in retriever/config.py.
