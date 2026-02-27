from __future__ import annotations

def calculate_state_dim(rl_cfg: dict) -> int:
    """
    Centralized logic to calculate the input state dimension for the RL policy.
    Ensures main.py (training) and orchestrator.py (inference/online) match.
    """
    state_cfg = rl_cfg.get("state", {})
    
    n_rays     = int(state_cfg.get("ultrasonic_rays",    1))
    feat_dim   = int(state_cfg.get("visual_feature_dim", 128))
    depth_flat = int(state_cfg.get("depth_map_flat",     256))
    map_sz     = int(state_cfg.get("local_map_size",     7))
    
    extra = 0
    if state_cfg.get("include_velocity", True): extra += 3
    if state_cfg.get("include_heading",  True): extra += 2
    
    return n_rays + feat_dim + depth_flat + map_sz**2 + extra
