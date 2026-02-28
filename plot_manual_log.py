import re
import matplotlib.pyplot as plt
import os
import sys

# We prefer the manual log if it exists, as requested by user
log_file = 'logs/manual_log.txt'
output_file = 'training_plot_1000.png'

episodes = []
rewards = []

if not os.path.exists(log_file):
    print(f"Skipping {log_file} (not found)")
    sys.exit(1)
    
print(f"Parsing {log_file}...")
try:
    with open(log_file, 'r', errors='ignore') as f:
        for line in f:
            if "ep=" in line and "R=" in line:
                # Format: ... ep=868 ... R=66.7 ...
                parts = line.split()
                ep_val = None
                rew_val = None
                
                for p in parts:
                    if p.startswith('ep='):
                        # handle ep=10/20 or ep=868
                        try:
                            ep_str = p.split('=')[1].replace('/',' ').split()[0]
                            ep_val = int(ep_str)
                        except: pass
                    if p.startswith('R='):
                        try:
                            rew_val = float(p.split('=')[1])
                        except: pass
                
                if ep_val is not None and rew_val is not None:
                    episodes.append(ep_val)
                    rewards.append(rew_val)
except Exception as e:
    print(f"Error reading {log_file}: {e}")

if not episodes:
    print("No episodes found.")
    sys.exit(0)

# Sort by episode number
zipped = sorted(zip(episodes, rewards))
episodes, rewards = zip(*zipped)

print(f"Found {len(episodes)} episodes.")
print(f"Range: {min(episodes)} - {max(episodes)}")

plt.figure(figsize=(12, 6))

# Plot raw data
plt.plot(episodes, rewards, label='Episode Reward', alpha=0.5, color='blue')

# Moving Average
if len(rewards) > 10:
    ma = []
    window = 10
    # Calculate MA based on the sorted list
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        chunk = rewards[start:i+1]
        ma.append(sum(chunk) / len(chunk))
        
    plt.plot(episodes, ma, label=f'Moving Avg ({window})', color='red', linewidth=2)

plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.title('Training Progress (Episodes 868-1000)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_file)
print(f"Saved plot to {output_file}")
