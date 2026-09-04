"""PPO optimizer: generative transformer synthesis of fixed-length designs.

Mirrors the reference PPO loop (batched sampling, actor/critic updates, target-KL
early stop) [4][2], but the actor emits a full gene vector in one pass. Uses
weight-conditioned scalarization [4][2] and the graded total_violation as a
feasibility gradient in the reward [4]. torch is imported here so the prototype
still runs random search + GA without it. Returns the shared optimizer contract [8].
"""
import numpy as np

from micard_design_optimization.utils.pareto import pareto_progress


def run_ppo_optimization(problem, epochs=40, mini_batch_size=32,
                         params=None, rng=None, session=None):
    import torch  # local import keeps torch optional
    from micard_design_optimization.optimization.ppo_architecture import Actor, Critic

    rng = rng or np.random.default_rng()
    params = params or {"learning_rate": 3e-4, "clip_ratio": 0.2,
                        "target_kl": 0.02, "update_iterations": 10,
                        "constraint_penalty": 1.0}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"PPO using device: {device}")

    num_objectives = problem.num_objectives
    min_mask = np.array([mm.lower() == "min" for mm in problem.objective_min_max])

    actor = Actor(device, params, problem.design_space, num_objectives).to(device)
    critic = Critic(device, params, num_objectives).to(device)

    all_des, all_obj = [], []
    all_constraints, all_constraint_vals = [], []
    all_actor_loss, all_critic_loss, all_kl, all_reward = [], [], [], []

    def scalar_reward(objectives, is_constrained, total_violation, weights):
        """Weighted-sum reward [4][2]; graded penalty for infeasibility [4]."""
        obj = np.array(objectives, dtype=float)
        # Flip minimization objectives to "higher is better" for the weighted sum.
        signed = np.where(min_mask, -obj, obj)
        reward = float(np.dot(weights, signed))
        if is_constrained:
            reward -= params["constraint_penalty"] * total_violation  # gradient toward feasible
        return reward

    for epoch in range(epochs):
        batch_weights, batch_genes, batch_logprobs, batch_rewards = [], [], [], []

        for _ in range(mini_batch_size):
            # Random preference weights, normalized to sum to 1 [4][2].
            w = rng.random(num_objectives)
            w = w / w.sum()

            genes, log_prob = actor.sample_action(w)
            objectives, is_constrained, cvals = problem.evaluate(genes)
            total_violation = float(np.sum(cvals))
            reward = scalar_reward(objectives, is_constrained,
                                   total_violation, w)

            batch_weights.append(w)
            batch_genes.append(genes)
            batch_logprobs.append(float(log_prob.item()))
            batch_rewards.append(reward)

            all_des.append(genes)
            all_obj.append(objectives)
            all_constraints.append(is_constrained)
            all_constraint_vals.append(cvals)
            if session and not is_constrained:
                session.save_design(problem.last_design)

        # Advantages = reward - baseline value(weights) [4].
        values = np.array([critic.value(w).item() for w in batch_weights])
        returns = np.array(batch_rewards, dtype=float)
        advantages = returns - values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor update with target-KL early stop [4][2].
        actor_loss, kl = 0.0, 0.0
        for _ in range(params["update_iterations"]):
            actor_loss, kl = actor.ppo_update(
                batch_weights, batch_genes, batch_logprobs, advantages)
            if kl > params["target_kl"]:
                break

        # Critic update [4][2].
        critic_loss = critic.ppo_update(batch_weights, returns)

        all_actor_loss.append(actor_loss)
        all_critic_loss.append(critic_loss)
        all_kl.append(kl)
        all_reward.append(float(np.mean(batch_rewards)))

        if epoch % max(1, epochs // 10) == 0:
            avg_r = float(np.mean(batch_rewards))
            print(f"PPO epoch {epoch}/{epochs} | reward {avg_r:.3f} | "
                  f"kl {kl:.4f} | actor {actor_loss:.4f} | critic {critic_loss:.4f}")

        # Log intermediate training stats for traceability [9].
        if session:
            session.log_intermediate(f"ppo_epoch_{epoch}", {
                "avg_reward": float(np.mean(batch_rewards)),
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "kl": kl,
            })

    # --- Post-processing: Pareto front + hypervolume over all evaluations ---
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)

    pareto_front_obj, hypervolumes = pareto_progress(
        all_obj, all_constraints, problem.objective_min_max
    )

    n_valid = int(np.sum(all_constraints == False))
    print(f"PPO: {n_valid} feasible designs of {len(all_obj)} evaluations.")

    # Optional training-curve logging (actor/critic/KL/reward) [4][2].
    if session:
        session.log_intermediate("ppo_training_curves", {
            "actor_loss": all_actor_loss,
            "critic_loss": all_critic_loss,
            "kl": all_kl,
            "avg_reward": all_reward,
        })

    return {
        "all_des": all_des,
        "all_obj": all_obj,
        "all_constraints": all_constraints,
        "pareto_front_obj": pareto_front_obj,
        "hypervolumes": hypervolumes,
        "num_objectives": num_objectives,
    }