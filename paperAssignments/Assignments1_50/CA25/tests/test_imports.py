def test_imports():
    import importlib
    import pkgutil

    # Ensure package importable
    import src
    from src import config, data, model, losses, utils

    assert pkgutil.find_loader('src') is not None
    assert hasattr(config, 'load_config')
    assert hasattr(data, 'get_dataloader')
    assert hasattr(model, 'MLP')
    assert hasattr(losses, 'regression_loss')
    assert hasattr(utils, 'set_seed')
