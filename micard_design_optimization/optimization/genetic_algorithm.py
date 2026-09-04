"""Custom multi-objective GA (NSGA-II style) — no pygad/pymoo dependency.

Operates on the fixed-length + active-length gene vector so it handles the
variable-length design space gracefully [design_repair 2][user]. Uses non-
dominated sorting + crowding distance (like the reference nsga2 selection
[genetic_algorithm 3]) and the custom hypervolume [genetic_algorithm 3].
"""
import numpy as np

from micard_design_optimization.utils.pareto import is_pareto_efficient, pareto_progress


def _dominates(a, b, min_mask):
    """True if a dominates b (minimization where min_mask, else maximization) [genetic_algorithm 3]."""
    better = np.where(min_mask, a < b, a > b)
    worse = np.where(min_mask, a > b, a < b)
    return np.any(better) and not np.any(worse)

def _constrained_dominates(a, b, viol_a, viol_b, min_mask, tol=1e-9):
    """Deb constrained-domination [standard NSGA-II constraint handling].

    - If both feasible: normal Pareto dominance.
    - If one feasible, one not: the feasible one dominates.
    - If both infeasible: the one with smaller total violation dominates.
    """
    a_feasible = viol_a <= tol
    b_feasible = viol_b <= tol

    if a_feasible and not b_feasible:
        return True
    if b_feasible and not a_feasible:
        return False
    if not a_feasible and not b_feasible:
        return viol_a < viol_b

    # Both feasible -> objective dominance (min where min_mask, else max) [3].
    better = np.where(min_mask, a < b, a > b)
    worse = np.where(min_mask, a > b, a < b)
    return np.any(better) and not np.any(worse)

def _fast_nondominated_sort(objs, min_mask, constraint_vals):
    """Return a list of fronts (each a list of indices)."""
    n = len(objs)
    S = [[] for _ in range(n)]
    ndom = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _constrained_dominates(objs[p], objs[q], constraint_vals[p], constraint_vals[q], min_mask):
                S[p].append(q)
            elif _constrained_dominates(objs[q], objs[p], constraint_vals[q], constraint_vals[p], min_mask):
                ndom[p] += 1
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding_distance(objs):
    """Standard NSGA-II crowding distance for a set of objective vectors."""
    n, m = objs.shape
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(objs[:, k])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = objs[order[-1], k] - objs[order[0], k]
        if span == 0:
            continue
        for i in range(1, n - 1):
            dist[order[i]] += (objs[order[i + 1], k] - objs[order[i - 1], k]) / span
    return dist


def _random_gene(var, rng):
    if var["type"] == "continuous":
        return rng.uniform(var["range"][0], var["range"][1])
    return rng.choice(np.array(var["range"]))


def _init_population(gene_space, pop_size, rng):
    return [np.array([_random_gene(v, rng) for v in gene_space], dtype=float)
            for _ in range(pop_size)]


def _crossover(p1, p2, rng):
    mask = rng.random(len(p1)) < 0.5
    return np.where(mask, p1, p2)


def _mutate(genes, gene_space, rng, rate=0.1):
    child = genes.copy()
    for i, var in enumerate(gene_space):
        if rng.random() < rate:
            child[i] = _random_gene(var, rng)
    return child


def run_genetic_algorithm(problem, pop_size=50, n_gen=40, rng=None, session=None):
    """Run the custom NSGA-II GA on the fixed-length gene space.

    Args mirror the reference GA's pop_size / n_gen roles [genetic_algorithm 3].
    """
    rng = rng or np.random.default_rng()
    gene_space = problem.design_space
    objective_min_max = problem.objective_min_max
    min_mask = np.array([mm.lower() == "min" for mm in objective_min_max])

    def evaluate(genes):
        objectives, is_constrained, constraint_vals = problem.evaluate(genes)
        # Penalize infeasible designs so they sort to the back [genetic_algorithm 3].
        obj = np.array(objectives, dtype=float)
        if is_constrained:
            obj = np.array(objectives) + np.sum(constraint_vals)
        return obj, is_constrained, constraint_vals

    population = _init_population(gene_space, pop_size, rng)
    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []

    pop_obj = []
    for genes in population:
        obj, con, cvals = evaluate(genes)
        pop_obj.append(obj)
        all_des.append(genes); all_obj.append(obj)
        all_constraints.append(con); all_constraint_vals.append(cvals)
        if session and not con:
            session.save_design(problem.last_design)
    pop_obj = np.array(pop_obj)

    for gen in range(n_gen):
        # --- Offspring via tournament on (rank, crowding) ---
        fronts = _fast_nondominated_sort(pop_obj, min_mask, all_constraints)
        rank = np.zeros(len(population), dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r

        def tournament():
            i, j = rng.integers(0, len(population), size=2)
            return i if rank[i] <= rank[j] else j

        offspring = []
        for _ in range(pop_size):
            p1, p2 = population[tournament()], population[tournament()]
            child = _mutate(_crossover(p1, p2, rng), gene_space, rng)
            offspring.append(child)

        off_obj = []
        for genes in offspring:
            obj, con, cvals = evaluate(genes)
            off_obj.append(obj)
            all_des.append(genes); all_obj.append(obj)
            all_constraints.append(con); all_constraint_vals.append(cvals)
            if session and not con:
                session.save_design(problem.last_design)
        off_obj = np.array(off_obj)

        # --- Environmental selection (mu + lambda) ---
        combined = population + offspring
        combined_obj = np.vstack([pop_obj, off_obj])
        fronts = _fast_nondominated_sort(combined_obj, min_mask, all_constraints)

        new_pop, new_obj = [], []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(combined[idx]); new_obj.append(combined_obj[idx])
            else:
                # Fill remaining slots by crowding distance.
                front_obj = combined_obj[front]
                cd = _crowding_distance(front_obj)
                order = np.argsort(-cd)
                for k in order[: pop_size - len(new_pop)]:
                    new_pop.append(combined[front[k]])
                    new_obj.append(combined_obj[front[k]])
                break
        population, pop_obj = new_pop, np.array(new_obj)

        if gen % max(1, n_gen // 10) == 0:
            print(f"GA generation {gen}/{n_gen}")

    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    pareto_front_obj, hypervolumes = pareto_progress(
        all_obj, all_constraints, objective_min_max
    )

    return {
        "all_des": all_des,
        "all_obj": all_obj,
        "all_constraints": all_constraints,
        "pareto_front_obj": pareto_front_obj,
        "hypervolumes": hypervolumes,
        "num_objectives": problem.num_objectives,
    }