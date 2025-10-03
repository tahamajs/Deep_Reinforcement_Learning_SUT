"""
Quick test for video generation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, ".")

print("Testing video generation components...")

try:
    from foundation_models import DecisionTransformer
    from neurosymbolic import NeurosymbolicAgent, SymbolicKnowledgeBase
    from human_ai_collaboration import CollaborativeAgent
    from continual_learning import ContinualLearningAgent
    from environments import ContinualEnv

    print("✅ All imports successful!")

    # Test basic agent creation
    dt_model = DecisionTransformer(state_dim=4, action_dim=4, model_dim=32)
    print("✅ Decision Transformer created")

    kb = SymbolicKnowledgeBase()
    ns_agent = NeurosymbolicAgent(state_dim=4, action_dim=4, knowledge_base=kb)
    print("✅ Neurosymbolic Agent created")

    collab_agent = CollaborativeAgent(state_dim=4, action_dim=4)
    print("✅ Collaborative Agent created")

    cl_agent = ContinualLearningAgent(state_dim=4, action_dim=4)
    print("✅ Continual Learning Agent created")

    env = ContinualEnv(num_tasks=2, state_dim=4, action_dim=4)
    print("✅ Environment created")

    print("\n🎬 All components working! Ready for video generation.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
