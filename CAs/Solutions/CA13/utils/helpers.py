import os
import torch
import numpy as np
import random
from typing import Dict, Any


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get appropriate device (CUDA if available)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_directory_structure(base_path: str = "."):
    """Create directory structure for CA13 project"""
    directories = [
        "agents",
        "models", 
        "environments",
        "buffers",
        "evaluation",
        "utils",
        "experiments",
        "results",
        "figures"
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        
        # Create __init__.py files
        init_file = os.path.join(full_path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {directory.title()} module\n")
    
    print("Directory structure created successfully!")


def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: torch.nn.Module, filepath: str, optimizer: torch.optim.Optimizer = None):
    """Save model and optionally optimizer state"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(model: torch.nn.Module, filepath: str, optimizer: torch.optim.Optimizer = None):
    """Load model and optionally optimizer state"""
    checkpoint = torch.load(filepath, map_location=get_device())
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath}")
    return model


def compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute gradient norm of model parameters"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def clip_grad_norm(model: torch.nn.Module, max_norm: float = 1.0):
    """Clip gradients to max norm"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = "step", 
                    **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler"""
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"


def print_model_summary(model: torch.nn.Module, input_shape: tuple = None):
    """Print model summary"""
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Model structure
    print(f"\nModel structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {module.__class__.__name__} ({param_count:,} parameters)")
    
    if input_shape is not None:
        print(f"\nInput shape: {input_shape}")
        try:
            # Test forward pass
            dummy_input = torch.randn(1, *input_shape)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error testing forward pass: {e}")
    
    print(f"{'='*60}\n")


def validate_environment(env, num_steps: int = 100):
    """Validate environment functionality"""
    print(f"Validating environment: {env}")
    
    try:
        obs = env.reset()
        print(f"✓ Reset successful. Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            if step == 0:
                print(f"✓ Step successful. Action space: {env.action_space}")
                print(f"  Reward range: [{reward}, {reward}]")
            
            if done:
                obs = env.reset()
                break
        
        print("✓ Environment validation successful!")
        
    except Exception as e:
        print(f"✗ Environment validation failed: {e}")


def setup_logging(log_file: str = "training.log", level: str = "INFO"):
    """Setup logging configuration"""
    import logging
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    print(f"Logging setup complete. Log file: {log_file}")


def create_config_template() -> Dict[str, Any]:
    """Create configuration template for experiments"""
    config = {
        "experiment": {
            "name": "ca13_experiment",
            "description": "CA13 Advanced RL Experiment",
            "seed": 42,
            "device": "auto"
        },
        "environment": {
            "name": "CartPole-v1",
            "max_steps": 500,
            "render": False
        },
        "agent": {
            "type": "DQN",
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 500,
            "hidden_dim": 128
        },
        "training": {
            "num_episodes": 200,
            "batch_size": 32,
            "buffer_size": 10000,
            "target_update_freq": 100,
            "eval_interval": 20,
            "eval_episodes": 10
        },
        "world_model": {
            "latent_dim": 32,
            "hidden_dim": 128,
            "learning_rate": 1e-3
        },
        "multi_agent": {
            "n_agents": 3,
            "enable_communication": True,
            "coordination_reward": 0.1
        }
    }
    
    return config


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to file"""
    import json
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from file"""
    import json
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from {filepath}")
    return config


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get GPU memory usage information"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        
        memory_info[f"gpu_{i}"] = {
            "allocated": memory_allocated,
            "reserved": memory_reserved,
            "total": memory_total,
            "free": memory_total - memory_reserved
        }
    
    return memory_info


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")


def benchmark_agent(agent, env, num_episodes: int = 100):
    """Benchmark agent performance"""
    import time
    
    start_time = time.time()
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(500):  # Max steps
            action = agent.act(obs, epsilon=0.0)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    benchmark_results = {
        "total_time": total_time,
        "episodes_per_second": num_episodes / total_time,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards)
    }
    
    print(f"Benchmark Results ({num_episodes} episodes):")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Episodes/second: {benchmark_results['episodes_per_second']:.2f}")
    print(f"  Mean reward: {benchmark_results['mean_reward']:.2f} ± {benchmark_results['std_reward']:.2f}")
    print(f"  Reward range: [{benchmark_results['min_reward']:.2f}, {benchmark_results['max_reward']:.2f}]")
    
    return benchmark_results
