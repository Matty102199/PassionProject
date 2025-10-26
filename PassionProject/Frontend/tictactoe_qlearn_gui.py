import random
import tkinter as tk
from collections import defaultdict
from tkinter import messagebox

# ---------------------------
# Game / RL core
# ---------------------------
EMPTY, X, O = " ", "X", "O"
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def winner_of(state: str):
    b = state
    for a, b2, c in WIN_LINES:
        if b[a] != EMPTY and b[a] == b[b2] == b[c]:
            return b[a]
    return None

def legal_moves(state: str):
    return [i for i, ch in enumerate(state) if ch == EMPTY]

def terminal(state: str):
    return winner_of(state) is not None or EMPTY not in state

def step(state: str, action: int, player: str):
    assert state[action] == EMPTY
    ns = state[:action] + player + state[action+1:]
    w = winner_of(ns)
    done = (w is not None) or (EMPTY not in ns)
    return ns, w, done

class QAgent:
    def __init__(self, mark, alpha=0.5, gamma=0.9, eps=0.2):
        self.mark = mark
        self.Q = defaultdict(float)   # (state, action) -> value
        self.alpha, self.gamma = alpha, gamma
        self.eps = eps

    def select(self, state):
        acts = legal_moves(state)
        if not acts:
            return None
        if random.random() < self.eps:
            return random.choice(acts)
        # greedy
        qvals = [(self.Q[(state, a)], a) for a in acts]
        return max(qvals)[1]

    def update(self, s, a, r, s2, done):
        key = (s, a)
        if done:
            target = r
        else:
            acts2 = legal_moves(s2)
            target = r + self.gamma * (max([self.Q[(s2, a2)] for a2 in acts2]) if acts2 else 0.0)
        self.Q[key] += self.alpha * (target - self.Q[key])

def self_play_train(episodes=40000, alpha=0.5, gamma=0.9, eps_start=0.2, eps_end=0.01):
    """Train two agents (X and O) by self-play; return trained agents."""
    ax = QAgent(X, alpha, gamma, eps_start)
    ao = QAgent(O, alpha, gamma, eps_start)

    def eps_decay(ep):
        t = ep / episodes
        return eps_end + (eps_start - eps_end) * (1 - t) ** 2

    for ep in range(1, episodes + 1):
        ax.eps = ao.eps = eps_decay(ep)
        s = EMPTY * 9
        player = X
        last = {X: None, O: None}

        while True:
            agent = ax if player == X else ao
            a = agent.select(s)
            if a is None:
                break  # should not happen in normal play

            s2, w, done = step(s, a, player)

            if not done:
                agent.update(s, a, 0.0, s2, done=False)
                last[player] = (s, a)
                s = s2
                player = O if player == X else X
                continue

            # terminal updates
            if w == player:
                r_self, r_other = 1.0, -1.0
            elif w is None:
                r_self, r_other = 0.0, 0.0
            else:
                r_self, r_other = -1.0, 1.0

            agent.update(s, a, r_self, s2, done=True)
            opp = O if player == X else X
            if last[opp] is not None:
                s_opp, a_opp = last[opp]
                agent_opp = ao if opp == O else ax
                agent_opp.update(s_opp, a_opp, r_other, s2, done=True)
            break

    # default to greedy for play time
    ax.eps = 0.0
    ao.eps = 0.0
    return ax, ao

# ---------------------------
# GUI
# ---------------------------
CELL = 120
PAD = 8
BOARD_SIZE = 3
CANVAS_SIZE = CELL * BOARD_SIZE

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        root.title("Q-Learning Tic-Tac-Toe")

        # Training status
        self.status = tk.StringVar(value="Training AI (quick)...")
        self.info = tk.Label(root, textvariable=self.status, font=("Segoe UI", 12))
        self.info.pack(pady=(6, 0))

        # Canvas
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", highlightthickness=1)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        # Controls
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0,10))
        self.side_var = tk.StringVar(value="X")  # human plays as X by default
        self.side_btn = tk.Button(btn_frame, text="Play as O (AI first)", command=self.toggle_side)
        self.side_btn.grid(row=0, column=0, padx=4)

        self.reset_btn = tk.Button(btn_frame, text="New Game", command=self.reset_game)
        self.reset_btn.grid(row=0, column=1, padx=4)

        self.retrain_btn = tk.Button(btn_frame, text="Retrain (fast)", command=self.retrain)
        self.retrain_btn.grid(row=0, column=2, padx=4)

        # Initialize state
        self.state = EMPTY * 9
        self.turn = X
        self.game_over = False
        self.trained_X = None
        self.trained_O = None

        # Draw empty board now
        self.draw_board()
        self.draw_marks()

        # Train once at startup, then start
        root.after(100, self.initial_train)

    # --- Training helpers ---
    def initial_train(self):
        self.status.set("Training AI…")
        self.root.update_idletasks()
        # Do a quick training run
        self.trained_X, self.trained_O = self.safe_train(episodes=40000)
        self.status.set("Ready: Click to play. You are X by default.")
        # If human chose O, AI should move first
        if self.side_var.get() == "O":
            self.ai_move_if_needed()

    def retrain(self):
        if messagebox.askyesno("Retrain", "Retrain the AI now? (Takes a moment)"):
            self.status.set("Retraining AI…")
            self.root.update_idletasks()
            self.trained_X, self.trained_O = self.safe_train(episodes=50000)
            self.status.set("Retrained. New Game started.")
            self.reset_game()

    def safe_train(self, episodes=40000):
        # Tiny guard in case user clicks retrain very early
        try:
            ax, ao = self_play_train(episodes=episodes)
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            # Fallback: minimal training to keep GUI usable
            ax, ao = self_play_train(episodes=2000)
        return ax, ao

    # --- Rendering ---
    def draw_board(self):
        self.canvas.delete("grid")
        for i in range(1, BOARD_SIZE):
            # vertical
            x = i * CELL
            self.canvas.create_line(x, 0, x, CANVAS_SIZE, width=3, tags="grid")
            # horizontal
            y = i * CELL
            self.canvas.create_line(0, y, CANVAS_SIZE, y, width=3, tags="grid")

    def draw_marks(self):
        self.canvas.delete("mark")
        for idx, ch in enumerate(self.state):
            r, c = divmod(idx, 3)
            x0, y0 = c * CELL + PAD, r * CELL + PAD
            x1, y1 = (c+1) * CELL - PAD, (r+1) * CELL - PAD
            if ch == X:
                # Draw X
                self.canvas.create_line(x0, y0, x1, y1, width=6, tags="mark")
                self.canvas.create_line(x0, y1, x1, y0, width=6, tags="mark")
            elif ch == O:
                # Draw O
                cx, cy = c * CELL + CELL/2, r * CELL + CELL/2
                radius = (CELL - 2*PAD) / 2
                self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, width=6, tags="mark")

    def set_state(self, new_state):
        self.state = new_state
        self.draw_marks()

    # --- Game control ---
    def toggle_side(self):
        if self.side_var.get() == "X":
            self.side_var.set("O")
            self.side_btn.config(text="Play as X (You first)")
            self.status.set("You are O. AI goes first.")
        else:
            self.side_var.set("X")
            self.side_btn.config(text="Play as O (AI first)")
            self.status.set("You are X. You go first.")
        self.reset_game()

    def reset_game(self):
        self.state = EMPTY * 9
        self.turn = X
        self.game_over = False
        self.draw_marks()
        self.status.set(f"New game. You are {self.side_var.get()}.")
        self.ai_move_if_needed()

    def ai_agent_for(self, mark):
        return self.trained_X if mark == X else self.trained_O

    def on_click(self, event):
        if self.game_over or self.trained_X is None:
            return
        # Determine cell
        c = int(event.x // CELL)
        r = int(event.y // CELL)
        if c < 0 or c > 2 or r < 0 or r > 2:
            return
        idx = r * 3 + c

        # Human move only when it's human's turn
        human_mark = self.side_var.get()
        if self.turn != human_mark:
            return
        if self.state[idx] != EMPTY:
            return

        # Apply human move
        s2, w, done = step(self.state, idx, self.turn)
        self.set_state(s2)
        if done:
            self.finish_game(w)
            return

        # Switch turn → AI
        self.turn = O if self.turn == X else X
        self.root.after(100, self.ai_move_if_needed)

    def ai_move_if_needed(self):
        if self.game_over or self.trained_X is None:
            return
        ai_mark = self.turn
        human_mark = self.side_var.get()
        if ai_mark == human_mark:
            return  # it's human's turn
        # AI acts greedily
        agent = self.ai_agent_for(ai_mark)
        agent.eps = 0.0
        a = agent.select(self.state)
        if a is None or self.state[a] != EMPTY:
            return
        s2, w, done = step(self.state, a, ai_mark)
        self.set_state(s2)
        if done:
            self.finish_game(w)
            return
        self.turn = O if self.turn == X else X

    def finish_game(self, w):
        self.game_over = True
        if w is None:
            self.status.set("Draw! Click New Game to play again.")
        else:
            self.status.set(f"{w} wins! Click New Game to play again.")

# ---------------------------
# Boot
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
