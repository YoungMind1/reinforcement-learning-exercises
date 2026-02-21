import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================
#  CONFIG
# ================================
COST_PER_STEP = 1.0
COST_PER_TURN = 100  # bigger → prefers straight lines
DISCOUNT = 1.0  # deterministic shortest path, so 1.0 is fine
MAX_ITER = 5000  # value iteration limit
TOL = 1e-6  # convergence threshold


# Directions: (dx, dy), and direction indices
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ["N", "E", "S", "W"]


# ================================
#  LOADING THE MAZE
# ================================
def load_maze(csv_file):
    grid = pd.read_csv(csv_file, header=None).values

    # identify start and goal
    # README: 200 is source, 100 is destination
    start = tuple(np.argwhere(grid == 200)[0])
    goal = tuple(np.argwhere(grid == 100)[0])

    # only treat:
    # 1, 2, 100, 200 as valid (traversable)
    # everything else is wall (=0)
    grid = np.where(np.isin(grid, [1, 2, 100, 200]), 1, 0)

    return grid, start, goal


# ================================
#  VALID MOVES
# ================================
def neighbors(r, c, grid):
    rows, cols = grid.shape
    for d_idx, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
            yield (nr, nc, d_idx)


# ================================
#  RL-STYLE VALUE ITERATION
#  V[state][direction]
#
#  direction stores the direction we entered from:
#   –1 means "no previous direction"
# ================================
def run_value_iteration(grid, start, goal, *, return_deltas=False):
    rows, cols = grid.shape

    # 3D Value table:
    # V[r][c][dir]  dir ∈ {-1, 0,1,2,3}
    # index 4 in dimension is used for dir = -1 (undefined)
    V = np.full((rows, cols, 5), np.inf)

    # Goal has value zero regardless of direction
    for d in range(5):
        V[goal][d] = 0

    deltas = []
    iters = 0
    for it in range(MAX_ITER):
        delta = 0

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:
                    continue

                for prev_dir in range(-1, 4):
                    V_old = V[r, c, prev_dir_to_idx(prev_dir)]

                    # Skip terminal
                    if (r, c) == goal:
                        continue

                    best = np.inf
                    for nr, nc, new_dir in neighbors(r, c, grid):
                        # compute cost
                        cost = COST_PER_STEP
                        if prev_dir != -1 and prev_dir != new_dir:
                            cost += COST_PER_TURN

                        val = cost + V[nr, nc, new_dir]

                        if val < best:
                            best = val

                    V[r, c, prev_dir_to_idx(prev_dir)] = best
                    if np.isfinite(V_old) and np.isfinite(best):
                        delta = max(delta, abs(V_old - best))

        deltas.append(delta)
        iters = it + 1
        if delta < TOL:
            break

    if return_deltas:
        return V, iters, deltas
    return V


def prev_dir_to_idx(d):
    return 4 if d == -1 else d


# ================================
#  FOLLOW THE POLICY (GREEDY)
# ================================
def extract_path(V, grid, start, goal):
    path = []
    r, c = start
    prev_dir = -1

    while (r, c) != goal:
        path.append(((r, c), DIR_NAMES[prev_dir] if prev_dir != -1 else "Start"))

        best = None
        best_val = np.inf

        for nr, nc, new_dir in neighbors(r, c, grid):
            cost = COST_PER_STEP
            if prev_dir != -1 and prev_dir != new_dir:
                cost += COST_PER_TURN

            val = cost + V[nr, nc, new_dir]
            if val < best_val:
                best_val = val
                best = (nr, nc, new_dir)

        if best is None:
            raise RuntimeError("Stuck: no path found even though states had value.")

        r, c, prev_dir = best

    path.append(((r, c), "GOAL"))
    return path


# ================================
#  PLOTTING
# ================================
def plot_path(grid, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r")

    ys = [p[0][0] for p in path]
    xs = [p[0][1] for p in path]

    plt.plot(xs, ys, linewidth=3)
    plt.scatter([xs[0]], [ys[0]], c="green", s=200, marker="o", label="Start")
    plt.scatter([xs[-1]], [ys[-1]], c="red", s=200, marker="x", label="Goal")
    plt.grid()
    plt.legend()
    plt.title("RL/DP Optimal Path")
    plt.show()


# ================================
#  MAIN
# ================================
if __name__ == "__main__":
    grid, start, goal = load_maze("matrix_path.csv")

    V = run_value_iteration(grid, start, goal)
    path = extract_path(V, grid, start, goal)

    steps = len(path) - 1
    turns = sum(1 for i in range(2, len(path)) if path[i - 1][1] != path[i][1])

    total_cost = steps * COST_PER_STEP + turns * COST_PER_TURN

    print("======== PATH FOUND (DP / RL) ========")
    print("Steps :", steps)
    print("Turns :", turns)
    print("Total Cost:", total_cost)
    print("\nFirst 10:")
    for i, p in enumerate(path[:10]):
        print(f"{i + 1}. {p}")

    plot_path(grid, path)
