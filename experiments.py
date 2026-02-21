import json
import math
import random
import time
from dataclasses import dataclass

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ["N", "E", "S", "W"]


def prev_dir_to_idx(d: int) -> int:
    return 4 if d == -1 else d


@dataclass(frozen=True)
class Maze:
    grid: np.ndarray  # 1 passable, 0 wall
    start: tuple[int, int]
    goal: tuple[int, int]
    raw: np.ndarray  # original csv values


def load_maze(csv_file: str) -> Maze:
    raw = pd.read_csv(csv_file, header=None).values
    start = tuple(int(x) for x in np.argwhere(raw == 200)[0])
    goal = tuple(int(x) for x in np.argwhere(raw == 100)[0])
    grid = np.where(np.isin(raw, [1, 2, 100, 200]), 1, 0).astype(np.int8)
    return Maze(grid=grid, start=start, goal=goal, raw=raw)


def neighbors(r: int, c: int, grid: np.ndarray):
    rows, cols = grid.shape
    for a, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
            yield nr, nc, a


def step_cost(
    prev_dir: int, new_dir: int, *, cost_step: float, cost_turn: float
) -> float:
    cost = cost_step
    if prev_dir != -1 and prev_dir != new_dir:
        cost += cost_turn
    return cost


def path_cost(
    path_dirs: list[int], *, cost_step: float, cost_turn: float
) -> tuple[int, int, float]:
    turns = 0
    for i in range(1, len(path_dirs)):
        if path_dirs[i - 1] != path_dirs[i]:
            turns += 1
    steps = len(path_dirs)
    total = steps * cost_step + turns * cost_turn
    return steps, turns, float(total)


def dijkstra_optimal(maze: Maze, *, cost_step: float, cost_turn: float):
    """Exact shortest path on expanded state space: (r,c,prev_dir_idx)."""
    grid = maze.grid
    sr, sc = maze.start
    gr, gc = maze.goal

    rows, cols = grid.shape
    inf = 1e30
    dist = np.full((rows, cols, 5), inf, dtype=np.float64)
    parent = {}  # (r,c,prev_idx) -> ((pr,pc,pprev_idx), action_dir)

    import heapq

    start_state = (sr, sc, 4)
    dist[start_state] = 0.0
    pq = [(0.0, *start_state)]
    popped = 0

    while pq:
        d, r, c, pidx = heapq.heappop(pq)
        popped += 1
        if d != dist[r, c, pidx]:
            continue
        if (r, c) == (gr, gc):
            # reached goal with some prev_dir
            break

        prev_dir = -1 if pidx == 4 else pidx
        for nr, nc, a in neighbors(r, c, grid):
            nd = d + step_cost(prev_dir, a, cost_step=cost_step, cost_turn=cost_turn)
            nstate = (nr, nc, a)
            if nd < dist[nstate]:
                dist[nstate] = nd
                parent[nstate] = ((r, c, pidx), a)
                heapq.heappush(pq, (nd, *nstate))

    # pick best prev_dir at goal
    best_pidx = int(np.argmin(dist[gr, gc, :]))
    best_cost = float(dist[gr, gc, best_pidx])
    if best_cost >= inf / 2:
        raise RuntimeError("No path found by Dijkstra.")

    # reconstruct actions
    actions = []
    cur = (gr, gc, best_pidx)
    while cur != start_state:
        if cur not in parent:
            raise RuntimeError("Failed to reconstruct optimal path.")
        prev, a = parent[cur]
        actions.append(a)
        cur = prev
    actions.reverse()

    return {
        "optimal_cost": best_cost,
        "optimal_actions": actions,
        "popped": popped,
    }


def value_iteration_cost_to_goal(
    maze: Maze,
    *,
    cost_step: float,
    cost_turn: float,
    discount: float = 1.0,
    max_iter: int = 5000,
    tol: float = 1e-6,
):
    grid = maze.grid
    rows, cols = grid.shape
    gr, gc = maze.goal

    V = np.zeros((rows, cols, 5), dtype=np.float64)
    V[gr, gc, :] = 0.0
    deltas = []

    for it in range(max_iter):
        delta = 0.0
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:
                    continue
                if (r, c) == (gr, gc):
                    continue
                for prev_dir in (-1, 0, 1, 2, 3):
                    pidx = prev_dir_to_idx(prev_dir)
                    old = V[r, c, pidx]
                    best = np.inf
                    for nr, nc, a in neighbors(r, c, grid):
                        cost = step_cost(
                            prev_dir, a, cost_step=cost_step, cost_turn=cost_turn
                        )
                        val = cost + discount * V[nr, nc, a]
                        if val < best:
                            best = val
                    V[r, c, pidx] = best
                    if np.isfinite(old) and np.isfinite(best):
                        delta = max(delta, abs(old - best))
        deltas.append(float(delta))
        if delta < tol:
            return V, (it + 1), deltas
    return V, max_iter, deltas


def greedy_path_from_V(
    maze: Maze,
    V: np.ndarray,
    *,
    cost_step: float,
    cost_turn: float,
    max_len: int | None = None,
):
    grid = maze.grid
    r, c = maze.start
    prev_dir = -1
    actions = []
    seen = set()
    if max_len is None:
        max_len = int(grid.size * 4)

    while (r, c) != maze.goal:
        if len(actions) >= max_len or (r, c, prev_dir) in seen:
            return None
        seen.add((r, c, prev_dir))

        best = None
        best_val = np.inf
        for nr, nc, a in neighbors(r, c, grid):
            cost = step_cost(prev_dir, a, cost_step=cost_step, cost_turn=cost_turn)
            val = cost + V[nr, nc, prev_dir_to_idx(a)]
            if val < best_val:
                best_val = val
                best = (nr, nc, a)
        if best is None:
            return None
        nr, nc, a = best
        r, c, prev_dir = nr, nc, a
        actions.append(a)
    return actions


def epsilon_greedy_action(r, c, prev_dir, grid, Q, epsilon, rng: random.Random):
    moves = list(neighbors(r, c, grid))
    if not moves:
        return None
    if rng.random() < epsilon:
        return rng.choice(moves)[2]

    pidx = prev_dir_to_idx(prev_dir)
    best_a = None
    best_q = -1e30
    for _, _, a in moves:
        q = float(Q[r, c, pidx, a])
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


def greedy_path_from_Q(
    maze: Maze, Q: np.ndarray, *, cost_step: float, cost_turn: float, max_len=None
):
    grid = maze.grid
    r, c = maze.start
    prev_dir = -1
    actions = []
    seen = set()
    if max_len is None:
        max_len = int(grid.size * 4)
    while (r, c) != maze.goal:
        if len(actions) >= max_len or (r, c, prev_dir) in seen:
            return None
        seen.add((r, c, prev_dir))

        pidx = prev_dir_to_idx(prev_dir)
        moves = list(neighbors(r, c, grid))
        if not moves:
            return None
        best_a = None
        best_q = -1e30
        for nr, nc, a in moves:
            q = float(Q[r, c, pidx, a])
            if q > best_q:
                best_q = q
                best_a = a
        if best_a is None:
            return None
        # take action
        for nr, nc, a in moves:
            if a == best_a:
                actions.append(a)
                r, c, prev_dir = nr, nc, a
                break
    return actions


def mc_control_Q(
    maze: Maze,
    *,
    episodes: int,
    epsilon: float,
    cost_step: float,
    cost_turn: float,
    discount: float,
    max_episode_len: int,
    seed: int,
    exploring_starts: bool = True,  # start from random passable cells
):
    """
    Monte Carlo control with exploring starts for better coverage.

    Key improvements for sparse-reward mazes:
    1. Exploring starts: randomly sample starting positions near goal sometimes
    2. Higher initial exploration with slower annealing
    3. First-visit MC update
    4. Only update from episodes that reach goal (pure MC)
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    grid = maze.grid
    rows, cols = grid.shape
    gr, gc = maze.goal

    # Build list of all passable cells, sorted by distance to goal
    passable = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 1]

    # Sort by Manhattan distance to goal (bias towards starting near goal for faster learning)
    passable_by_dist = sorted(passable, key=lambda p: abs(p[0] - gr) + abs(p[1] - gc))
    n_passable = len(passable_by_dist)

    # Q-values: maximize return (negative cost); initialize optimistically
    Q = np.zeros((rows, cols, 5, 4), dtype=np.float64)
    N = np.zeros((rows, cols, 5, 4), dtype=np.float64) + 1e-9

    costs: list[float] = []
    reached: list[bool] = []

    last_cost = float("inf")
    last_reached = False
    eval_interval = 10

    successful_episodes = 0

    for ep in range(episodes):
        # Exploring starts strategy:
        # - 50% start from actual start (important for learning the actual policy)
        # - 30% start from cells close to goal (curriculum learning)
        # - 20% start from random cells
        roll = rng.random()
        if not exploring_starts or roll < 0.5:
            r, c = maze.start
            prev_dir = -1
        elif roll < 0.8:
            # Sample from cells closer to goal (first half of sorted list)
            idx = rng.randint(0, n_passable // 2)
            r, c = passable_by_dist[idx]
            prev_dir = rng.choice([-1, 0, 1, 2, 3])
        else:
            r, c = rng.choice(passable)
            prev_dir = rng.choice([-1, 0, 1, 2, 3])

        traj = []  # (r,c,pidx,a,reward)
        reached_goal = False

        # Higher exploration early, slower decay
        progress = min(1.0, ep / max(1, 0.9 * episodes))
        eps_ep = epsilon + (1.0 - epsilon) * (1.0 - progress)

        for _ in range(max_episode_len):
            if (r, c) == maze.goal:
                reached_goal = True
                break

            a = epsilon_greedy_action(r, c, prev_dir, grid, Q, eps_ep, rng)
            if a is None:
                break
            rew = -step_cost(prev_dir, a, cost_step=cost_step, cost_turn=cost_turn)

            # step
            nr = r + DIRS[a][0]
            nc = c + DIRS[a][1]
            if not (0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1):
                break

            traj.append((r, c, prev_dir_to_idx(prev_dir), a, float(rew)))
            r, c, prev_dir = nr, nc, a

        # Only update from successful episodes (pure MC)
        if traj and reached_goal:
            successful_episodes += 1
            returns = [0.0] * len(traj)
            G = 0.0
            for i in range(len(traj) - 1, -1, -1):
                rew = traj[i][4]
                G = rew + discount * G
                returns[i] = float(G)

            # First-visit MC update
            seen = set()
            for (sr, sc, spidx, a, _rew), Gt in zip(traj, returns, strict=False):
                key = (sr, sc, spidx, a)
                if key in seen:
                    continue
                seen.add(key)
                N[sr, sc, spidx, a] += 1.0
                Q[sr, sc, spidx, a] += (Gt - Q[sr, sc, spidx, a]) / N[sr, sc, spidx, a]

        # eval greedy policy cost (not every episode)
        if ep % eval_interval == 0:
            actions = greedy_path_from_Q(
                maze, Q, cost_step=cost_step, cost_turn=cost_turn
            )
            if actions is None:
                last_cost = float("inf")
                last_reached = False
            else:
                _steps, _turns, last_cost = path_cost(
                    actions, cost_step=cost_step, cost_turn=cost_turn
                )
                last_reached = True

        costs.append(last_cost)
        reached.append(last_reached)

    return Q, {
        "cost": costs,
        "reached": reached,
        "successful_episodes": successful_episodes,
    }


def q_learning(
    maze: Maze,
    *,
    episodes: int,
    alpha: float,
    epsilon: float,
    cost_step: float,
    cost_turn: float,
    discount: float,
    max_episode_len: int,
    seed: int,
):
    """
    Q-learning: off-policy TD control.
    Uses max over next actions for the TD target (greedy).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    grid = maze.grid
    rows, cols = grid.shape
    Q = np.zeros((rows, cols, 5, 4), dtype=np.float64)

    costs: list[float] = []
    reached: list[bool] = []

    last_cost = float("inf")
    last_reached = False
    eval_interval = 10

    for _ep in range(episodes):
        r, c = maze.start
        prev_dir = -1

        for _ in range(max_episode_len):
            if (r, c) == maze.goal:
                break
            a = epsilon_greedy_action(r, c, prev_dir, grid, Q, epsilon, rng)
            if a is None:
                break

            rew = -step_cost(prev_dir, a, cost_step=cost_step, cost_turn=cost_turn)
            nr = r + DIRS[a][0]
            nc = c + DIRS[a][1]
            if not (0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1):
                break

            spidx = prev_dir_to_idx(prev_dir)
            if (nr, nc) == maze.goal:
                target = rew
            else:
                nspidx = prev_dir_to_idx(a)
                # max over available moves at next state (off-policy)
                nmoves = list(neighbors(nr, nc, grid))
                if not nmoves:
                    target = rew
                else:
                    best_next = max(float(Q[nr, nc, nspidx, na]) for _, _, na in nmoves)
                    target = rew + discount * best_next
            Q[r, c, spidx, a] += alpha * (target - Q[r, c, spidx, a])

            r, c, prev_dir = nr, nc, a

        if _ep % eval_interval == 0:
            actions = greedy_path_from_Q(
                maze, Q, cost_step=cost_step, cost_turn=cost_turn
            )
            if actions is None:
                last_cost = float("inf")
                last_reached = False
            else:
                _steps, _turns, last_cost = path_cost(
                    actions, cost_step=cost_step, cost_turn=cost_turn
                )
                last_reached = True

        costs.append(last_cost)
        reached.append(last_reached)

    return Q, {"cost": costs, "reached": reached}


def sarsa(
    maze: Maze,
    *,
    episodes: int,
    alpha: float,
    epsilon: float,
    cost_step: float,
    cost_turn: float,
    discount: float,
    max_episode_len: int,
    seed: int,
):
    """
    SARSA: on-policy TD control.
    Uses the actual next action (from ε-greedy policy) for the TD target.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    grid = maze.grid
    rows, cols = grid.shape
    Q = np.zeros((rows, cols, 5, 4), dtype=np.float64)

    costs: list[float] = []
    reached: list[bool] = []

    last_cost = float("inf")
    last_reached = False
    eval_interval = 10

    for _ep in range(episodes):
        r, c = maze.start
        prev_dir = -1

        # Choose initial action ε-greedily
        a = epsilon_greedy_action(r, c, prev_dir, grid, Q, epsilon, rng)

        for _ in range(max_episode_len):
            if (r, c) == maze.goal or a is None:
                break

            rew = -step_cost(prev_dir, a, cost_step=cost_step, cost_turn=cost_turn)
            nr = r + DIRS[a][0]
            nc = c + DIRS[a][1]
            if not (0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1):
                break

            spidx = prev_dir_to_idx(prev_dir)

            # Choose next action ε-greedily (for SARSA target)
            next_a = None
            if (nr, nc) != maze.goal:
                next_a = epsilon_greedy_action(nr, nc, a, grid, Q, epsilon, rng)

            if (nr, nc) == maze.goal:
                target = rew
            elif next_a is None:
                target = rew  # terminal (no valid actions)
            else:
                nspidx = prev_dir_to_idx(a)
                # Use the actual next action (on-policy)
                target = rew + discount * float(Q[nr, nc, nspidx, next_a])

            Q[r, c, spidx, a] += alpha * (target - Q[r, c, spidx, a])

            r, c, prev_dir = nr, nc, a
            a = next_a

        if _ep % eval_interval == 0:
            actions = greedy_path_from_Q(
                maze, Q, cost_step=cost_step, cost_turn=cost_turn
            )
            if actions is None:
                last_cost = float("inf")
                last_reached = False
            else:
                _steps, _turns, last_cost = path_cost(
                    actions, cost_step=cost_step, cost_turn=cost_turn
                )
                last_reached = True

        costs.append(last_cost)
        reached.append(last_reached)

    return Q, {"cost": costs, "reached": reached}


def first_stable_hit(
    costs: list[float], *, threshold: float, window: int
) -> int | None:
    good = [c <= threshold for c in costs]
    for i in range(len(good) - window + 1):
        if all(good[i : i + window]):
            return i + 1  # 1-based episode index
    return None


def plot_learning_curves(
    curves: dict[str, list[float]], out_png: str, *, optimal_cost: float
):
    plt.figure(figsize=(9, 4.8))
    for name, costs in curves.items():
        y = np.array([c if math.isfinite(c) else np.nan for c in costs], dtype=float)
        # rolling mean (window 200)
        w = 200
        y2 = pd.Series(y).rolling(window=w, min_periods=1).mean().to_numpy()
        plt.plot(y2, label=name, linewidth=2)
    plt.axhline(
        optimal_cost,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Optimal (Dijkstra)",
    )
    plt.xlabel("Episode")
    plt.ylabel("Greedy policy cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_vi_deltas(deltas: list[float], out_png: str):
    plt.figure(figsize=(7.2, 4.2))
    plt.semilogy(np.maximum(np.array(deltas), 1e-16), linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Delta (max |V-V_old|)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_maze_with_path(maze: Maze, actions: list[int], out_png: str):
    grid = maze.grid
    r, c = maze.start
    ys = [r]
    xs = [c]
    for a in actions:
        r += DIRS[a][0]
        c += DIRS[a][1]
        ys.append(r)
        xs.append(c)

    plt.figure(figsize=(7.0, 7.0))
    plt.imshow(1 - grid, cmap="gray", interpolation="nearest")
    plt.plot(xs, ys, linewidth=3, color="#1f77b4")
    plt.scatter([xs[0]], [ys[0]], c="#2ca02c", s=160, marker="o", label="Start")
    plt.scatter([xs[-1]], [ys[-1]], c="#d62728", s=180, marker="x", label="Goal")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def ensure_dir(path: str):
    import os

    os.makedirs(path, exist_ok=True)


def main():
    maze = load_maze("matrix_path.csv")
    ensure_dir("fig")
    ensure_dir("out")

    # Experiment configs
    cost_step = 1.0
    turn_costs = [0.0, 10.0, 100.0]
    episodes = 12000  # reasonable number for TD methods
    episodes_mc = (
        30000  # more episodes for MC (still won't converge but shows learning)
    )
    max_episode_len = 2000
    seeds = [0, 1, 2]
    epsilon = 0.15
    alpha = 0.10
    discount = 1.0
    stable_window = 150
    stable_eps = 1.01  # within 1% of optimal

    summary = {
        "settings": {
            "episodes_td": episodes,
            "episodes_mc": episodes_mc,
            "max_episode_len": max_episode_len,
            "seeds": seeds,
            "epsilon": epsilon,
            "alpha": alpha,
            "stable_window": stable_window,
            "stable_eps": stable_eps,
        },
        "cases": [],
    }

    for cost_turn in turn_costs:
        case = {"cost_turn": cost_turn}

        # Optimal (exact)
        t0 = time.time()
        opt = dijkstra_optimal(maze, cost_step=cost_step, cost_turn=cost_turn)
        opt_time = time.time() - t0
        optimal_cost = opt["optimal_cost"]
        case["optimal"] = {
            "cost": optimal_cost,
            "steps": len(opt["optimal_actions"]),
            "turns": path_cost(
                opt["optimal_actions"], cost_step=cost_step, cost_turn=cost_turn
            )[1],
            "popped": opt["popped"],
            "time_sec": opt_time,
        }
        plot_maze_with_path(
            maze, opt["optimal_actions"], f"fig/path_opt_turn{int(cost_turn)}.png"
        )

        # Value iteration (sanity vs optimal)
        t0 = time.time()
        V, iters, deltas = value_iteration_cost_to_goal(
            maze,
            cost_step=cost_step,
            cost_turn=cost_turn,
            discount=1.0,
            max_iter=5000,
            tol=1e-6,
        )
        vi_time = time.time() - t0
        vi_actions = greedy_path_from_V(
            maze, V, cost_step=cost_step, cost_turn=cost_turn
        )
        if vi_actions is None:
            vi_cost = float("inf")
            vi_steps = None
            vi_turns = None
        else:
            vi_steps, vi_turns, vi_cost = path_cost(
                vi_actions, cost_step=cost_step, cost_turn=cost_turn
            )
            plot_maze_with_path(
                maze, vi_actions, f"fig/path_vi_turn{int(cost_turn)}.png"
            )
        plot_vi_deltas(deltas, f"fig/vi_delta_turn{int(cost_turn)}.png")
        case["value_iteration"] = {
            "iters": iters,
            "final_delta": deltas[-1] if deltas else None,
            "cost": vi_cost,
            "steps": vi_steps,
            "turns": vi_turns,
            "time_sec": vi_time,
        }

        # RL runs (multiple seeds)
        td_curves = []
        sarsa_curves = []
        mc_curves = []
        td_hits = []
        sarsa_hits = []
        mc_hits = []
        td_final_costs = []
        sarsa_final_costs = []
        mc_final_costs = []
        td_best_costs = []
        sarsa_best_costs = []
        mc_best_costs = []
        timings_list = []

        Qtd_seed0 = None
        Qsarsa_seed0 = None
        Qmc_seed0 = None

        print(f"Running experiments for cost_turn={cost_turn}...")

        for s in seeds:
            print(f"  Seed {s}...")

            # Q-learning (off-policy TD)
            t0 = time.time()
            _Qtd, log_td = q_learning(
                maze,
                episodes=episodes,
                alpha=alpha,
                epsilon=epsilon,
                cost_step=cost_step,
                cost_turn=cost_turn,
                discount=discount,
                max_episode_len=max_episode_len,
                seed=s,
            )
            td_time = time.time() - t0
            td_curves.append(log_td["cost"])
            hit_td = first_stable_hit(
                log_td["cost"],
                threshold=stable_eps * optimal_cost,
                window=stable_window,
            )
            td_hits.append(hit_td)
            td_final_costs.append(float(log_td["cost"][-1]))
            td_best_costs.append(
                float(np.nanmin(np.array(log_td["cost"], dtype=float)))
            )
            if s == seeds[0]:
                Qtd_seed0 = _Qtd

            # SARSA (on-policy TD)
            t0 = time.time()
            _Qsarsa, log_sarsa = sarsa(
                maze,
                episodes=episodes,
                alpha=alpha,
                epsilon=epsilon,
                cost_step=cost_step,
                cost_turn=cost_turn,
                discount=discount,
                max_episode_len=max_episode_len,
                seed=s,
            )
            sarsa_time = time.time() - t0
            sarsa_curves.append(log_sarsa["cost"])
            hit_sarsa = first_stable_hit(
                log_sarsa["cost"],
                threshold=stable_eps * optimal_cost,
                window=stable_window,
            )
            sarsa_hits.append(hit_sarsa)
            sarsa_final_costs.append(float(log_sarsa["cost"][-1]))
            sarsa_best_costs.append(
                float(np.nanmin(np.array(log_sarsa["cost"], dtype=float)))
            )
            if s == seeds[0]:
                Qsarsa_seed0 = _Qsarsa

            # MC control (with exploring starts) - more episodes
            t0 = time.time()
            _Qmc, log_mc = mc_control_Q(
                maze,
                episodes=episodes_mc,
                epsilon=epsilon,
                cost_step=cost_step,
                cost_turn=cost_turn,
                discount=discount,
                max_episode_len=max_episode_len,
                seed=s,
                exploring_starts=True,
            )
            mc_time = time.time() - t0
            mc_curves.append(log_mc["cost"])
            hit_mc = first_stable_hit(
                log_mc["cost"],
                threshold=stable_eps * optimal_cost,
                window=stable_window,
            )
            mc_hits.append(hit_mc)
            mc_final_costs.append(float(log_mc["cost"][-1]))
            mc_best_costs.append(
                float(np.nanmin(np.array(log_mc["cost"], dtype=float)))
            )
            if s == seeds[0]:
                Qmc_seed0 = _Qmc

            # keep per-seed timings for rough comparison
            timings_list.append(
                {
                    "seed": s,
                    "td_sec": td_time,
                    "sarsa_sec": sarsa_time,
                    "mc_sec": mc_time,
                }
            )

        case["timings"] = timings_list

        # aggregate curves: median for plots
        td_med = np.nanmedian(np.array(td_curves, dtype=float), axis=0).tolist()
        sarsa_med = np.nanmedian(np.array(sarsa_curves, dtype=float), axis=0).tolist()
        mc_med = np.nanmedian(np.array(mc_curves, dtype=float), axis=0).tolist()
        plot_learning_curves(
            {
                "Q-learning (off-policy)": td_med,
                "SARSA (on-policy)": sarsa_med,
                "MC control": mc_med,
            },
            f"fig/learning_turn{int(cost_turn)}.png",
            optimal_cost=optimal_cost,
        )

        def hit_stats(hits):
            finite = [h for h in hits if h is not None]
            return {
                "hits": hits,
                "hit_rate": float(len(finite) / len(hits)),
                "hit_median": int(np.median(finite)) if finite else None,
                "hit_min": int(min(finite)) if finite else None,
            }

        case["td"] = hit_stats(td_hits)
        case["sarsa"] = hit_stats(sarsa_hits)
        case["mc"] = hit_stats(mc_hits)
        case["td"].update(
            {
                "final_costs": td_final_costs,
                "final_cost_median": float(np.median(td_final_costs)),
                "best_costs": td_best_costs,
                "best_cost_median": float(np.median(td_best_costs)),
            }
        )
        case["sarsa"].update(
            {
                "final_costs": sarsa_final_costs,
                "final_cost_median": float(np.median(sarsa_final_costs)),
                "best_costs": sarsa_best_costs,
                "best_cost_median": float(np.median(sarsa_best_costs)),
            }
        )
        case["mc"].update(
            {
                "final_costs": mc_final_costs,
                "final_cost_median": float(np.median(mc_final_costs)),
                "best_costs": mc_best_costs,
                "best_cost_median": float(np.median(mc_best_costs)),
            }
        )

        if Qtd_seed0 is not None:
            td_actions = greedy_path_from_Q(
                maze, Qtd_seed0, cost_step=cost_step, cost_turn=cost_turn
            )
            if td_actions is not None:
                plot_maze_with_path(
                    maze, td_actions, f"fig/path_td_turn{int(cost_turn)}.png"
                )
        if Qsarsa_seed0 is not None:
            sarsa_actions = greedy_path_from_Q(
                maze, Qsarsa_seed0, cost_step=cost_step, cost_turn=cost_turn
            )
            if sarsa_actions is not None:
                plot_maze_with_path(
                    maze, sarsa_actions, f"fig/path_sarsa_turn{int(cost_turn)}.png"
                )
        if Qmc_seed0 is not None:
            mc_actions = greedy_path_from_Q(
                maze, Qmc_seed0, cost_step=cost_step, cost_turn=cost_turn
            )
            if mc_actions is not None:
                plot_maze_with_path(
                    maze, mc_actions, f"fig/path_mc_turn{int(cost_turn)}.png"
                )
        case["episodes_td"] = episodes
        case["episodes_mc"] = episodes_mc
        case["threshold"] = stable_eps * optimal_cost
        summary["cases"].append(case)
        print(
            f"  Done. TD hit_rate={case['td']['hit_rate']:.0%}, "
            f"SARSA hit_rate={case['sarsa']['hit_rate']:.0%}, "
            f"MC hit_rate={case['mc']['hit_rate']:.0%}"
        )

    with open("out/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nResults saved to out/summary.json")


if __name__ == "__main__":
    main()
