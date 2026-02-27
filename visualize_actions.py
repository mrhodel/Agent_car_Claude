import re
import os
import sys

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

LOG_FILE = "logs/agent.log"
OUTPUT_IMG = "logs/action_distribution.png"

def parse_logs(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Log file not found: {filepath}")
        return {}

    episode_data = {}
    line_pattern = re.compile(r"\[Train\] ep=(\d+)\s+actions\s+(.*)")
    
    with open(filepath, 'r') as f:
        for line in f:
            match = line_pattern.search(line)
            if match:
                ep_num = int(match.group(1))
                action_str = match.group(2)
                actions = {}
                parts = action_str.strip().split()
                for part in parts:
                    if ':' in part:
                        try:
                            name_val = part.split(':')
                            if len(name_val) == 2:
                                name, val = name_val
                                val = float(val.replace('%', ''))
                                actions[name] = val
                        except ValueError:
                            continue
                episode_data[ep_num] = actions
    return episode_data

def visualize(data, num_episodes=15):
    if not data:
        print("No action data found.")
        return

    sorted_eps = sorted(data.keys())
    recent_eps = sorted_eps[-num_episodes:]
    
    print(f"\n--- Action Distribution (Last {len(recent_eps)} Episodes) ---")
    
    all_actions = set()
    for ep in recent_eps:
        all_actions.update(data[ep].keys())
    sorted_actions = sorted(list(all_actions))
    
    # Header
    print(f"{'Ep':<4} | ", end="")
    for act in sorted_actions:
        abbr = act[:6]
        if 'rotate' in act: abbr = 'rot_' + act.split('_')[1][0].upper()
        elif 'strafe' in act: abbr = 'str_' + act.split('_')[1][0].upper()
        elif 'gimbal' in act: abbr = 'gimb'
        print(f"{abbr:<6}", end=" | ")
    print("\n" + "-" * 80)
    
    for ep in recent_eps:
        print(f"{ep:<4} | ", end="")
        for act in sorted_actions:
            val = data[ep].get(act, 0)
            s_val = f"{int(val)}"
            if val >= 20: 
                # Bold via ANSI
                print(f"\033[1m{s_val:<6}\033[0m", end=" | ") 
            else:
                print(f"{s_val:<6}", end=" | ")
        print()
    print("-" * 80)

if __name__ == "__main__":
    data = parse_logs(LOG_FILE)
    visualize(data)
