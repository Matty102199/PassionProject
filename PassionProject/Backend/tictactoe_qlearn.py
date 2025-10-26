import random
from collections import defaultdict

# --- Game helpers ---
EMPTY, X, O = " ", "X", "O"
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def winner_of(state: str):
    b = state
    for a,b2,c in WIN_LINES:
        if b[a] != EMPTY and b[a] == b[b2] == b[c]:
            return b[a]
    return None

def legal_moves(state: str):
    return [i for i,ch in enumerate(state) if ch == EMPTY]

def terminal(state: str):
    return winner_of(state) is not None or EMPTY not in state

def step(state: str, action: int, player: str):
    assert state[action] == EMPTY
    ns = state[:action] + player + state[action+1:]
    w = winner_of(ns)
    done = (w is not None) or (EMPTY not in ns)
    return ns, w, done

def print_board(state: str):
    b = [c if c != EMPTY else "." for c in state]
    rows = [" {} | {} | {} ".format(*b[r:r+3]) for r in (0,3,6)]
    print("\n" + "\n-----------\n".join(rows) + "\n")

# --- Q-learning agent ---
class QAgent:
    def __init__(self, mark, alpha=0.5, gamma=0.9, eps=0.2):
        self.mark = mark
        self.Q = defaultdict(float)   # keys: (state, action) -> value
        self.alpha, self.gamma = alpha, gamma
        self.eps = eps

    def select(self, state):
        acts = legal_moves(state)
        if not acts: return None
        if random.random() < self.eps:
            return random.choice(acts)
        # greedy:
        qvals = [(self.Q[(state, a)], a) for a in acts]
        return max(qvals)[1]

    def update(self, s, a, r, s2, done):
        key = (s, a)
        if done:
            target = r
        else:
            acts2 = legal_moves(s2)
            target = r + self.gamma * (max([self.Q[(s2,a2)] for a2 in acts2]) if acts2 else 0.0)
        self.Q[key] += self.alpha * (target - self.Q[key])

# --- Training by self-play ---
def self_play_train(episodes=50000, alpha=0.5, gamma=0.9, eps_start=0.2, eps_end=0.01):
    ax = QAgent(X, alpha, gamma, eps_start)
    ao = QAgent(O, alpha, gamma, eps_start)

    def eps_decay(ep):
        # cosine-ish smooth decay (tweak freely)
        t = ep / episodes
        return eps_end + (eps_start - eps_end) * (1 - t) ** 2

    for ep in range(1, episodes+1):
        ax.eps = ao.eps = eps_decay(ep)
        s = EMPTY*9
        player = X
        last = {X: None, O: None}  # store (s, a) for final terminal update

        while True:
            agent = ax if player == X else ao
            a = agent.select(s)
            # Safety: in degenerate cases (shouldn't happen), break
            if a is None: break

            s2, w, done = step(s, a, player)
            # rewards from the perspective of the acting agent
            if not done:
                r = 0.0
                agent.update(s, a, r, s2, done=False)
                last[player] = (s, a)
                s = s2
                player = O if player == X else X
                continue

            # Terminal: assign rewards
            if w == player:          # the mover just won
                r_self, r_other = 1.0, -1.0
            elif w is None:          # draw
                r_self, r_other = 0.0, 0.0
            else:                     # should not occur here
                r_self, r_other = -1.0, 1.0

            agent.update(s, a, r_self, s2, done=True)
            # also update the opponent's last move with its terminal reward
            opp = O if player == X else X
            if last[opp] is not None:
                s_opp, a_opp = last[opp]
                # opponent receives the opposite terminal reward
                agent_opp = ao if opp == O else ax
                agent_opp.update(s_opp, a_opp, r_other, s2, done=True)
            break

    return ax, ao

# --- Baselines / evaluation ---
def random_policy(state):
    acts = legal_moves(state)
    return random.choice(acts) if acts else None

def eval_vs_random(agent, episodes=5000, as_mark=X):
    wins = draws = losses = 0
    for _ in range(episodes):
        s = EMPTY*9
        player = X
        while True:
            if player == as_mark:
                a = agent.select(s)  # agent acts greedily if we set eps=0
            else:
                a = random_policy(s)
            if a is None: break
            s, w, done = step(s, a, player)
            if done:
                if w == as_mark: wins += 1
                elif w is None: draws += 1
                else: losses += 1
                break
            player = O if player == X else X
    return wins, draws, losses

# --- Human play ---
def play_human_vs_agent(agent_as=O):
    # agent plays as O by default; set agent_as=X for first move AI
    agent = trained_O if agent_as == O else trained_X
    agent.eps = 0.0  # act greedily when playing human
    s = EMPTY*9
    player = X
    print("Board positions are 0..8:")
    print(" 0 | 1 | 2 \n-----------\n 3 | 4 | 5 \n-----------\n 6 | 7 | 8 ")
    while True:
        print_board(s)
        if player == agent_as:
            a = agent.select(s)
            print(f"AI ({player}) moves to", a)
        else:
            try:
                a = int(input(f"Your move as {player} (0-8): "))
            except Exception:
                print("Enter a number 0..8"); continue
            if a not in range(9) or s[a] != EMPTY:
                print("Illegal move, try again.")
                continue
        s, w, done = step(s, a, player)
        if done:
            print_board(s)
            print("Result:", w if w else "Draw")
            break
        player = O if player == X else X

# --- Run training, evaluate, then let user play ---
if __name__ == "__main__":
    print("Training... (this is fast)")
    trained_X, trained_O = self_play_train(
        episodes=40000,  # bump to 200k+ for near-perfect play
        alpha=0.5, gamma=0.9, eps_start=0.2, eps_end=0.01
    )

    # Evaluate (greedy)
    trained_X.eps = trained_O.eps = 0.0
    wx, dx, lx = eval_vs_random(trained_X, 3000, as_mark=X)
    wo, do, lo = eval_vs_random(trained_O, 3000, as_mark=O)
    print(f"X-agent vs random: {wx}W {dx}D {lx}L (out of 3000)")
    print(f"O-agent vs random: {wo}W {do}D {lo}L (out of 3000)")

    # Play against the trained agent (toggle side below)
    play_human_vs_agent(agent_as=O)
