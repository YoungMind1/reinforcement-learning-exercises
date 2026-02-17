
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

EPISODES = 5000
MAX_EPISODE_LENGTH = 2000

EPSILON = 0.15     # exploration rate
SEED = 0           # for reproducibility
random.seed(SEED)
np.random.seed(SEED)


# Directions
DIRS = [(-1,0), (0,1), (1,0), (0,-1)]
DIR_NAMES = ["N", "E", "S", "W"]


def prev_dir_to_idx(d):
    return 4 if d == -1 else d


# ====================================
#  LOAD MAZE
# ====================================
def load_maze(csv_file):
    grid = pd.read_csv(csv_file, header=None).values

    start = tuple(np.argwhere(grid == 100)[0])
    goal  = tuple(np.argwhere(grid == 200)[0])

    # Only allow movement on:
    # 2, 100, 200
    # Everything else becomes wall (0)
    grid = np.where(np.isin(grid, [2,100,200]), 1, 0)
    return grid, start, goal


# ====================================
#  VALID NEIGHBORS
# ====================================
def neighbors(r, c, grid):
    rows, cols = grid.shape
    for d_idx, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
            yield nr, nc, d_idx


# ====================================
#  MONTE CARLO LEARNING
#  We store:
#  V[r][c][direction]
#  direction: -1 means "no previous direction"
# ====================================
def monte_carlo_control(grid, start, goal):

    rows, cols = grid.shape

    # Value function
    V = np.zeros((rows, cols, 5))

    # Counts for incremental mean updates
    N = np.zeros((rows, cols, 5)) + 1e-9

    for ep in range(EPISODES):

        # ===============================
        # Generate an episode
        # ===============================
        episode = []
        r, c = start
        prev_dir = -1

        for t in range(MAX_EPISODE_LENGTH):

            # If goal reached, episode ends
            if (r, c) == goal:
                episode.append(((r, c, prev_dir), 0))
                break

            # Choose action ε-greedily
            if random.random() < EPSILON:
                # explore
                possible_moves = list(neighbors(r, c, grid))
                nr, nc, new_dir = random.choice(possible_moves)
            else:
                # greedy wrt V
                best = None
                best_val = +1e9
                for nr, nc, new_dir in neighbors(r, c, grid):

                    step_cost = COST_PER_STEP
                    if prev_dir != -1 and prev_dir != new_dir:
                        step_cost += COST_PER_TURN

                    val = step_cost + V[nr, nc, new_dir]

                    if val < best_val:
                        best_val = val
                        best = (nr, nc, new_dir)

                nr, nc, new_dir = best

            # reward is negative cost
            reward = -(COST_PER_STEP + (COST_PER_TURN if prev_dir != -1 and prev_dir != new_dir else 0))

            # store transition
            episode.append(((r, c, prev_dir), reward))

            # move
            r, c, prev_dir = nr, nc, new_dir

        # ===============================
        # Monte Carlo return calculation
        # ===============================
        G = 0
        for (state, reward) in reversed(episode):
            (sr, sc, sd) = state
            G = reward + DISCOUNT * G

            idx = prev_dir_to_idx(sd)

            N[sr, sc, idx] += 1
            V[sr, sc, idx] += (G - V[sr, sc, idx]) / N[sr, sc, idx]

    return V


# ====================================
#  EXTRACT BEST PATH GREEDILY
# ====================================
def extract_path(V, grid, start, goal):

    r, c = start
    prev_dir = -1
    path = []

    while (r, c) != goal:
        path.append(((r,c), DIR_NAMES[prev_dir] if prev_dir != -1 else "Start"))

        best = None
        best_val = +1e9

        for nr, nc, new_dir in neighbors(r, c, grid):

            cost = COST_PER_STEP
            if prev_dir != -1 and prev_dir != new_dir:
                cost += COST_PER_TURN

            val = cost + V[nr, nc, new_dir]

            if val < best_val:
                best_val = val
                best = (nr, nc, new_dir)

        if best is None:
            raise RuntimeError("No path found through learned values.")

        r, c, prev_dir = best

    path.append(((r,c), "GOAL"))
    return path


# ====================================
#  PLOT
# ====================================
def plot_path(grid, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="binary")
    ys = [p[0][0] for p in path]
    xs = [p[0][1] for p in path]

    plt.plot(xs, ys, linewidth=3)
    plt.scatter([xs[0]],[ys[0]], c="green", s=200, marker="o", label="Start")
    plt.scatter([xs[-1]],[ys[-1]], c="red", s=200, marker="X", label="Goal")
    plt.grid()
    plt.legend()
    plt.title("Monte Carlo RL Path")
    plt.show()


# ====================================
#  MAIN
# ====================================
if __name__ == "__main__":

    grid, start, goal = load_maze("matrix_path.csv")
    

    V = monte_carlo_control(grid, start, goal)
    path = extract_path(V, grid, start, goal)

    steps = len(path) - 1
    turns = sum(1 for i in range(2, len(path))
                if path[i-1][1] != path[i][1])

    total_cost = steps*COST_PER_STEP + turns*COST_PER_TURN

    print("======== Monte Carlo RL Path ========")
    print("Steps :", steps)
    print("Turns :", turns)
    print("Total Cost:", total_cost)
    print("\nFirst 10:")
    for i,p in enumerate(path[:10]):
        print(f"{i+1}. {p}")

    plot_path(grid, path)
