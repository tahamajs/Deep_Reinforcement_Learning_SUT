#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all module imports"""
    print("Testing CA16 module imports...")
    
    try:
        # Test foundation models
        from foundation_models import DecisionTransformer, FoundationModelTrainer, ScalingAnalyzer
        print("✓ Foundation models imported successfully")
        
        # Test neurosymbolic
        from neurosymbolic import NeurosymbolicAgent, SymbolicKnowledgeBase
        print("✓ Neurosymbolic modules imported successfully")
        
        # Test human-AI collaboration
        from human_ai_collaboration import CollaborativeAgent
        print("✓ Human-AI collaboration imported successfully")
        
        # Test continual learning
        from continual_learning import ContinualLearningAgent
        print("✓ Continual learning imported successfully")
        
        # Test environments
        from environments import SymbolicGridWorld, CollaborativeGridWorld
        print("✓ Environments imported successfully")
        
        # Test advanced computational (optional)
        try:
            from advanced_computational import QuantumInspiredRL, NeuromorphicNetwork
            print("✓ Advanced computational modules imported successfully")
        except ImportError as e:
            print(f"⚠ Advanced computational modules not available: {e}")
        
        # Test real-world deployment (optional)
        try:
            from real_world_deployment import ProductionRLSystem, SafetyMonitor
            print("✓ Real-world deployment modules imported successfully")
        except ImportError as e:
            print(f"⚠ Real-world deployment modules not available: {e}")
        
        print("\n🎉 All imports completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of imported classes"""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        import numpy as np
        
        # Test DecisionTransformer
        from foundation_models import DecisionTransformer
        dt = DecisionTransformer(state_dim=8, action_dim=4, model_dim=64, num_heads=4, num_layers=2)
        print("✓ DecisionTransformer instantiated successfully")
        
        # Test NeurosymbolicAgent
        from neurosymbolic import NeurosymbolicAgent, SymbolicKnowledgeBase
        kb = SymbolicKnowledgeBase()
        ns_agent = NeurosymbolicAgent(state_dim=8, action_dim=4, knowledge_base=kb)
        print("✓ NeurosymbolicAgent instantiated successfully")
        
        # Test CollaborativeAgent
        from human_ai_collaboration import CollaborativeAgent
        collab = CollaborativeAgent(state_dim=8, action_dim=4)
        print("✓ CollaborativeAgent instantiated successfully")
        
        # Test environments
        from environments import SymbolicGridWorld, CollaborativeGridWorld
        env1 = SymbolicGridWorld(size=6)
        env2 = CollaborativeGridWorld(size=6)
        print("✓ Environments instantiated successfully")
        
        print("🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("CA16 Module Import and Functionality Test")
    print("=" * 50)
    
    import_success = test_imports()
    if import_success:
        func_success = test_basic_functionality()
        
        if func_success:
            print("\n🎉 All tests passed! The CA16 modules are ready to use.")
            sys.exit(0)
        else:
            print("\n❌ Functionality tests failed.")
            sys.exit(1)
    else:
        print("\n❌ Import tests failed.")
        sys.exit(1)
