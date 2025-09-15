import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# -------------------- Toggle --------------------
show_overrides = False  # set True to highlight overrides, False for clean view

# -------------------- Parameters --------------------
generations = 400
width = generations * 2 + 1
# random rule
rule_number = np.random.randint(0, 256)
rule_number = 182        # <--- change this to any Wolfram rule (0–255)

# Initial conditions
random_initial =  True #: random initial row; False: start with single active cell
# random_initial = False
np.random.seed(21)
if random_initial:
    initial_row = np.random.randint(0, 2, width)
else:
    initial_row = np.zeros(width, dtype=int)
    initial_row[width // 2] = 1
init_desc = "random initial row" if random_initial else "single active cell"

# Memory approach parameters
m = 3
flip_threshold = 2
# d = 10  # majority over last (d+1) states
#  random d between 3 and 10
d = np.random.randint(3, 10)
d = 7

# -------------------- Rule helpers --------------------
def make_rule_array(rule_number):
    """Convert rule number (0–255) into array of 8 outputs for neighborhoods 111..000"""
    return np.array([(rule_number >> i) & 1 for i in range(7, -1, -1)], dtype=int)

RULE = make_rule_array(rule_number)

def apply_rule(left, center, right):
    """Apply Wolfram rule to neighborhood."""
    idx = (left << 2) | (center << 1) | right  # value 0–7
    return RULE[7 - idx]  # 7-idx to map 111→0 ... 000→7

def majority(bits_stack):
    s = bits_stack.sum(axis=0)
    return (s * 2 >= bits_stack.shape[0]).astype(int)

# -------------------- Update rules with mask --------------------
def next_row_baseline_with_mask(history):
    row = history[0]
    left = np.roll(row, 1)
    center = row
    right = np.roll(row, -1)
    nxt = apply_rule(left, center, right)
    overridden = np.zeros_like(nxt, dtype=int)
    return nxt, overridden

def next_row_volatility_majority_with_mask(history, m=3, flip_threshold=2, d=10):
    row = history[0]
    left = np.roll(row, 1)
    center = row.copy()
    right = np.roll(row, -1)

    if len(history) < max(m, d) + 1:
        nxt = apply_rule(left, center, right)
        overridden = np.zeros_like(nxt, dtype=int)
        return nxt, overridden

    # Volatility detection
    flips = [(history[j] != history[j+1]).astype(int) for j in range(m)]
    flips = np.stack(flips, axis=0)
    flip_counts = flips.sum(axis=0)
    trig = (flip_counts >= flip_threshold)

    # Majority memory over last d+1 states
    window = np.stack([history[j] for j in range(d + 1)], axis=0)
    mem_center = majority(window)

    eff_center = center.copy()
    eff_center[trig] = mem_center[trig]

    nxt = apply_rule(left, eff_center, right)
    overridden = trig.astype(int)
    return nxt, overridden

# -------------------- Runner --------------------
def run_with_masks(step_fn, generations, width, initial_row, hist_depth):
    states = np.zeros((generations, width), dtype=int)
    masks = np.zeros((generations, width), dtype=int)
    H = deque(maxlen=hist_depth)
    H.appendleft(initial_row.copy())
    for g in range(generations):
        states[g] = H[0]
        nxt, mask = step_fn(H)
        masks[g] = mask
        H.appendleft(nxt)
    return states, masks

# -------------------- Simulations --------------------
hist_depth = max(m, d) + 1
base_states, base_masks = run_with_masks(next_row_baseline_with_mask, generations, width, initial_row, hist_depth)
mem_states,  mem_masks  = run_with_masks(
    lambda H: next_row_volatility_majority_with_mask(H, m=m, flip_threshold=flip_threshold, d=d),
    generations, width, initial_row, hist_depth
)

# -------------------- Encode for display --------------------
if show_overrides:
    encoded_base = 2 * base_states + base_masks
    encoded_mem  = 2 * mem_states  + mem_masks
    vmin, vmax = 0, 3
else:
    encoded_base = base_states
    encoded_mem  = mem_states
    vmin, vmax = 0, 1

# -------------------- Plot --------------------
plt.figure(figsize=(12, 10))
plt.imshow(encoded_base, aspect='auto', interpolation='nearest', origin='upper', vmin=vmin, vmax=vmax)
plt.title(f"Rule {rule_number} (baseline) — {init_desc}, {generations} generations")
plt.xlabel("Cell index")
plt.ylabel("Generation (0 at top)")
plt.tight_layout()
plt.savefig(f"rule{rule_number}_baseline.png")

plt.figure(figsize=(12, 10))
plt.imshow(encoded_mem, aspect='auto', interpolation='nearest', origin='upper', vmin=vmin, vmax=vmax)
plt.title(f"Rule {rule_number} with memory (last {d+1}, m={m}, flip_th={flip_threshold}) — {init_desc}, {generations} generations")
plt.xlabel("Cell index")
plt.ylabel("Generation (0 at top)")
plt.tight_layout()
plt.savefig(f"rule{rule_number}_memory.png")
plt.savefig(f"latest_memory.png")
