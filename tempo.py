import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# ====================================
#  CONFIG
# ====================================
COST_PER_STEP = 1.0
COST_PER_TURN = 10.0

DISCOUNT = 1.0
ALPHA = 0.1
EPISODES = 5000
MAX_EPISODE_LENGTH = 2000
EPSILON = 0.1

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


# Directions (dx,dy)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ["N", "E", "S", "W"]


def prev_dir_to_idx(d):
    return 4 if d == -1 else d


# ====================================
#  LOAD MAZE
# ====================================
def load_maze(csv_file):
    grid = pd.read_csv(csv_file, header=None).values

    # README: 200 is source, 100 is destination
    start = tuple(np.argwhere(grid == 200)[0])
    goal = tuple(np.argwhere(grid == 100)[0])

    # Only let cells == 1,2,100,200 be passable
    grid = np.where(np.isin(grid, [1, 2, 100, 200]), 1, 0)

    return grid, start, goal


# ====================================
#  VALID ACTIONS
# ====================================
def neighbors(r, c, grid):
    rows, cols = grid.shape
    for d_idx, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
            yield nr, nc, d_idx


# ====================================
#  TD(0) LEARNING
#  V[r][c][direction]
# ====================================
def td_learning(grid, start, goal):
    rows, cols = grid.shape

    # Initialize V = 0
    V = np.zeros((rows, cols, 5))

    for ep in range(EPISODES):
        r, c = start
        prev_dir = -1  # we haven't moved yet

        for t in range(MAX_EPISODE_LENGTH):
            # If we hit the goal, episode ends immediately
            if (r, c) == goal:
                break

            # ====================================
            #  Choose action (ε-greedy)
            # ====================================
            moves = list(neighbors(r, c, grid))
            if not moves:
                break

            if random.random() < EPSILON:
                # explore
                nr, nc, new_dir = random.choice(moves)
            else:
                # greedy wrt V (maximize expected return)
                best = None
                best_val = -1e18

                for nr, nc, new_dir in moves:
                    step_cost = COST_PER_STEP
                    if prev_dir != -1 and prev_dir != new_dir:
                        step_cost += COST_PER_TURN

                    reward_eval = -step_cost
                    val = reward_eval + DISCOUNT * V[nr, nc, new_dir]

                    if val > best_val:
                        best_val = val
                        best = (nr, nc, new_dir)

                if best is None:
                    break
                nr, nc, new_dir = best

            # ====================================
            #  Reward
            #  (negative cost, like before)
            # ====================================
            reward = -(
                COST_PER_STEP
                + (COST_PER_TURN if prev_dir != -1 and prev_dir != new_dir else 0)
            )

            # ====================================
            #  TD TARGET
            # ====================================
            s_idx = prev_dir_to_idx(prev_dir)

            if (nr, nc) == goal:
                target = reward  # terminal
            else:
                target = reward + DISCOUNT * V[nr, nc, new_dir]

            # ====================================
            #  TD UPDATE
            # V(s) = V(s) + α * (target - V(s))
            # ====================================
            V[r, c, s_idx] += ALPHA * (target - V[r, c, s_idx])

            # Move
            r, c, prev_dir = nr, nc, new_dir

    return V


# ====================================
#  FOLLOW BEST POLICY GREEDILY
# ====================================
def extract_path(V, grid, start, goal):
    r, c = start
    prev_dir = -1
    path = []

    max_len = int(grid.size * 4)
    seen = set()

    while (r, c) != goal:
        if len(path) > max_len or (r, c, prev_dir) in seen:
            raise RuntimeError("Policy got stuck in a loop.")
        seen.add((r, c, prev_dir))

        path.append(((r, c), DIR_NAMES[prev_dir] if prev_dir != -1 else "Start"))

        best = None
        best_val = -1e18

        for nr, nc, new_dir in neighbors(r, c, grid):
            step_cost = COST_PER_STEP
            if prev_dir != -1 and prev_dir != new_dir:
                step_cost += COST_PER_TURN
            reward = -step_cost

            val = reward + DISCOUNT * V[nr, nc, new_dir]

            if val > best_val:
                best_val = val
                best = (nr, nc, new_dir)

        if best is None:
            raise RuntimeError("No valid move — policy failed.")

        r, c, prev_dir = best

    path.append(((r, c), "GOAL"))
    return path


# ====================================
#  PLOT PATH
# ====================================
def plot_path(grid, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="binary")

    ys = [p[0][0] for p in path]
    xs = [p[0][1] for p in path]

    plt.plot(xs, ys, linewidth=3)
    plt.scatter([xs[0]], [ys[0]], c="green", s=200, marker="o", label="Start")
    plt.scatter([xs[-1]], [ys[-1]], c="red", s=200, marker="X", label="Goal")
    plt.grid()
    plt.legend()
    plt.title("Temporal Difference RL Path")
    plt.show()


# ====================================
#  MAIN
# ====================================
if __name__ == "__main__":
    grid, start, goal = load_maze("matrix_path.csv")

    V = td_learning(grid, start, goal)  # ← core TD learner
    path = extract_path(V, grid, start, goal)  # ← follow greedy policy

    steps = len(path) - 1
    turns = sum(1 for i in range(2, len(path)) if path[i - 1][1] != path[i][1])

    total_cost = steps * COST_PER_STEP + turns * COST_PER_TURN

    print("======== Temporal Difference (TD) Result ========")
    print("Steps :", steps)
    print("Turns :", turns)
    print("Total Cost:", total_cost)
    print("\nFirst 10 moves:")
    for i, p in enumerate(path[:10]):
        print(f"{i + 1}. {p}")

    plot_path(grid, path)
