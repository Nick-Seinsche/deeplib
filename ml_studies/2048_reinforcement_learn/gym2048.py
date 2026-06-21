"""Game Logic and Visuals for 2048."""

import tkinter as tk
import numpy as np

# --------------------- Your Logic Functions ---------------------


def get_valid_actions(state):
    """
    Return the list of valid actions (0: up, 1: right, 2: down, 3: left).

    An action is valid if it modifies the board.
    """
    actions = []
    for a in range(4):
        new_state, modified, _ = _update_state(state, a)
        if modified:
            actions.append(a)
    return actions


def generate_board() -> np.array:
    """Generate a new game board with two random tiles (2 or 4)."""
    state = np.zeros((16,), dtype=np.int32)
    indices = np.random.choice(16, size=2, replace=False)
    state[indices[0]] = 1 if np.random.rand() < 0.9 else 2
    state[indices[1]] = 1 if np.random.rand() < 0.9 else 2
    return state


def update_board(state: np.array, action: int) -> tuple:
    """
    Apply the given action to the board. If the move is valid, insert a new tile.

    Returns the updated state, computed reward, and game-over status.
    """
    new_state, modified, reward = _update_state(state, action)

    if modified:
        zero_indices = np.where(new_state == 0)[0]
        if len(zero_indices) > 0:
            chosen_index = np.random.choice(zero_indices)
            new_state[chosen_index] = 1 + int(np.random.random() >= 0.9)

    done = not any(_update_state(np.copy(new_state), i)[1] for i in range(4))

    reward = np.log2(1 + reward)

    if np.max(new_state) > np.max(state):
        reward += 1

    if np.array_equal(state, new_state):
        reward -= 0.1  # valid move

    if done:
        reward -= 100.0  # discourage ending early

    return new_state, reward, done


def _update_state(state: np.array, action: int):
    """
    Simulate a move in the given direction (0-3).

    Returns the updated state, whether it was modified, and the reward gained.
    """
    modified = False
    reward = 0
    state_updated = np.zeros(16, dtype=np.int32)
    if action == 0:  # up
        for i in range(4):
            state_updated[[i + 0, i + 4, i + 8, i + 12]], m, r = _update_row_column(
                state[[i + 0, i + 4, i + 8, i + 12]]
            )
            modified |= m
            reward += r
    elif action == 1:  # right
        for i in range(4):
            s = [4 * i + j for j in [3, 2, 1, 0]]
            u, m, r = _update_row_column(state[s])
            state_updated[s] = u
            modified |= m
            reward += r
    elif action == 2:  # down
        for i in range(4):
            state_updated[[i + 12, i + 8, i + 4, i + 0]], m, r = _update_row_column(
                state[[i + 12, i + 8, i + 4, i + 0]]
            )
            modified |= m
            reward += r
    elif action == 3:  # left
        for i in range(4):
            s = [4 * i + j for j in range(4)]
            u, m, r = _update_row_column(state[s])
            state_updated[s] = u
            modified |= m
            reward += r
    return state_updated, modified, reward


def _update_row_column(arr: np.ndarray):
    """
    Merge a row or column according to 2048 game rules.

    Returns the updated array, whether it was modified, and reward for merges.
    """
    non_zero = arr[arr != 0]
    merged = []
    reward = 0
    skip = False
    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            merged_val = non_zero[i] + 1
            merged.append(merged_val)
            reward += 2 ** merged_val
            skip = True
        else:
            merged.append(non_zero[i])
    merged += [0] * (4 - len(merged))
    modified = not np.array_equal(arr, merged)
    return np.array(merged, dtype=np.int32), modified, reward

# --------------------- GUI + Game Integration ---------------------


class Game2048Board:
    """GUI class for rendering and interacting with a 2048 game board."""

    def __init__(self, master):
        """Initialize the game board and GUI."""
        self.grid_size = 4
        self.tile_size = 100
        self.state = generate_board()
        self.tiles = []
        self.bg_colors = {
            0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
            16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
            256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"
        }
        self.master = master
        if master is not None:
            self.master.title("2048 Game")
            self.setup_ui()
            self.render_board()
            self.master.bind("<Key>", self.handle_keypress)

    def setup_ui(self):
        """Create the visual grid layout using tkinter Frames and Labels."""
        self.canvas = tk.Frame(self.master, bg="#bbada0", bd=10)
        self.canvas.pack()
        for i in range(16):
            frame = tk.Frame(
                self.canvas,
                width=self.tile_size,
                height=self.tile_size,
                bg=self.bg_colors[0],
                bd=5
            )
            frame.grid(row=i // 4, column=i % 4, padx=5, pady=5)
            label = tk.Label(
                frame,
                text="",
                bg=self.bg_colors[0],
                fg="#776e65",
                font=("Helvetica", 24, "bold"),
                width=4,
                height=2
            )
            label.pack(expand=True, fill="both")
            self.tiles.append(label)

    def render_board(self):
        """Render the board state onto the GUI tiles."""
        for i in range(16):
            val = 2 ** self.state[i] if self.state[i] > 0 else 0
            self.tiles[i].config(
                text=str(val) if val else "",
                bg=self.bg_colors.get(val, "#3c3a32"),
                fg="#f9f6f2" if val > 4 else "#776e65"
            )

    def handle_keypress(self, event):
        """Handle keyboard input and apply moves based on arrow keys."""
        key_map = {"Up": 0, "Right": 1, "Down": 2, "Left": 3}
        if event.keysym not in key_map:
            return
        action = key_map[event.keysym]
        self.make_move(action=action)

    def make_move(self, action):
        new_state, reward, done = update_board(np.copy(self.state), action)
        if not np.array_equal(self.state, new_state):
            if self.master is not None:
                self.animate_move(self.state, new_state)
            self.state = new_state
            if self.master is not None:
                self.render_board()
            if done and self.master is not None:
                self.game_over()

    def let_model_play(self, model):
        valid_actions = get_valid_actions(self.state)

        import tensorflow as tf

        next_move = np.argmax(
            np.array(model(np.expand_dims(tf.one_hot(self.state, 12), axis=0)))[0]
            * [-np.inf if a not in valid_actions else 1 for a in range(4)])

        self.make_move(next_move)
        self.master.after(1000, self.let_model_play, model)

    def animate_move(self, old_state, new_state):
        """Animate tile changes when a move is made."""
        changed = [i for i in range(16) if old_state[i] != new_state[i]]
        for i in changed:
            self.tiles[i].config(bg="#ffcc00")
        self.master.update()
        self.master.after(100)

    def game_over(self):
        """Display game over screen."""
        print("Game Over!")
        if self.master is not None:
            for tile in self.tiles:
                tile.config(bg="#a39489")

# --------------------- Run ---------------------


if __name__ == "__main__":
    root = tk.Tk()
    board = Game2048Board(root)
    root.mainloop()

    # st = np.array([4, 16, 32, 8, 32, 128, 16, 4, 8, 64, 8, 32, 4, 8, 16, 4])
    # st = np.log2(st).astype(int)
    # print(update_board(st, 0))
