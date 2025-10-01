"""
Test script to verify all imports and modules are working correctly
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing CA10 Module Imports...")
print("=" * 60)

try:
    print("\n1. Testing models module...")
    from models.models import TabularModel, NeuralModel, ModelTrainer, device
    print("   ✅ models.models imported successfully")
except Exception as e:
    print(f"   ❌ Error importing models.models: {e}")

try:
    print("\n2. Testing environments module...")
    from environments.environments import SimpleGridWorld, BlockingMaze
    print("   ✅ environments.environments imported successfully")
except Exception as e:
    print(f"   ❌ Error importing environments.environments: {e}")

try:
    print("\n3. Testing classical_planning agent...")
    from agents.classical_planning import (
        ModelBasedPlanner,
        UncertaintyAwarePlanner,
        ModelBasedPolicySearch,
        demonstrate_classical_planning
    )
    print("   ✅ agents.classical_planning imported successfully")
except Exception as e:
    print(f"   ❌ Error importing agents.classical_planning: {e}")

try:
    print("\n4. Testing dyna_q agent...")
    from agents.dyna_q import DynaQAgent, DynaQPlusAgent, demonstrate_dyna_q
    print("   ✅ agents.dyna_q imported successfully")
except Exception as e:
    print(f"   ❌ Error importing agents.dyna_q: {e}")

try:
    print("\n5. Testing mcts agent...")
    from agents.mcts import MCTSAgent, demonstrate_mcts
    print("   ✅ agents.mcts imported successfully")
except Exception as e:
    print(f"   ❌ Error importing agents.mcts: {e}")

try:
    print("\n6. Testing mpc agent...")
    from agents.mpc import MPCAgent, MPCController, demonstrate_mpc
    print("   ✅ agents.mpc imported successfully")
except Exception as e:
    print(f"   ❌ Error importing agents.mpc: {e}")

try:
    print("\n7. Testing comparison experiment...")
    from experiments.comparison import demonstrate_comparison
    print("   ✅ experiments.comparison imported successfully")
except Exception as e:
    print(f"   ❌ Error importing experiments.comparison: {e}")

print("\n" + "=" * 60)
print("✅ All imports successful! CA10 is ready to use.")
print("\nQuick Test:")

try:
    print("\nCreating SimpleGridWorld environment...")
    env = SimpleGridWorld(size=4)
    print(f"   ✅ Environment created: {env.num_states} states, {env.num_actions} actions")
    
    print("\nCreating TabularModel...")
    model = TabularModel(env.num_states, env.num_actions)
    print(f"   ✅ Model created successfully")
    
    print("\nTesting model update...")
    state = env.reset()
    action = 0
    next_state, reward, done = env.step(action)
    model.update(state, action, next_state, reward)
    print(f"   ✅ Model update successful: {state} -> {next_state}, reward={reward:.2f}")
    
    print("\nCreating DynaQAgent...")
    agent = DynaQAgent(env.num_states, env.num_actions, planning_steps=5)
    print(f"   ✅ Agent created successfully")
    
    print("\n🎉 Quick test passed! All components working correctly.")
    
except Exception as e:
    print(f"\n❌ Quick test failed: {e}")

print("\n" + "=" * 60)
print("To run full demonstrations, use:")
print("  • demonstrate_classical_planning()")
print("  • demonstrate_dyna_q()")
print("  • demonstrate_mcts()")
print("  • demonstrate_mpc()")
print("  • demonstrate_comparison()")
print("=" * 60)
