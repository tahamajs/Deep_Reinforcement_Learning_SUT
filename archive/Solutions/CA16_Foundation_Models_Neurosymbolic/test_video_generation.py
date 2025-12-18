"""
Quick test for video generation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config

print("Testing video generation components...")

try:
    from src.foundation_models.algorithms import DecisionTransformer
    from src.neurosymbolic.policies import NeurosymbolicAgent
    from src.neurosymbolic.knowledge_base import SymbolicKnowledgeBase
    from src.human_ai_collaboration.collaborative_agent import CollaborativeAgent
    from src.continual_learning.continual_agent import ContinualLearningAgent
    from src.environments.continual_env import ContinualEnv

    print("✅ All imports successful!")

    # Test basic agent creation
    dt_model = DecisionTransformer(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM, model_dim=config.TRANSFORMER_DIM)
    print("✅ Decision Transformer created")

    kb = SymbolicKnowledgeBase()
    ns_agent = NeurosymbolicAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM, knowledge_base=kb)
    print("✅ Neurosymbolic Agent created")

    collab_agent = CollaborativeAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
    print("✅ Collaborative Agent created")

    cl_agent = ContinualLearningAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
    print("✅ Continual Learning Agent created")

    env = ContinualEnv(num_tasks=config.NUM_TASKS, state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
    print("✅ Environment created")

    print("\n🎬 All components working! Ready for video generation.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
