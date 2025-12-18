from data import RandomMDPDataset


def test_random_mdp_dataset_item():
    ds = RandomMDPDataset(num_transitions=10, obs_dim=3, action_dim=2)
    item = ds[0]
    assert set(item.keys()) == {"obs", "action", "reward", "done", "next_obs"}
    assert item["obs"].shape == (3,)
    assert isinstance(item["action"].item(), int)
    assert isinstance(float(item["reward"]), float)
    assert item["next_obs"].shape == (3,)
