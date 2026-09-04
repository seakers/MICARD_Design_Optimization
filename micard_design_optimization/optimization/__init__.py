"""Interchangeable search strategies -- random search, a custom NSGA-II
genetic algorithm, and transformer PPO. All three share the same
`evaluate()`/gene-vector contract (see utils.design_space) and return
contract (all_des, all_obj, all_constraints, pareto_front_obj,
hypervolumes, num_objectives), so run.run() can pick any subset of them.
"""
