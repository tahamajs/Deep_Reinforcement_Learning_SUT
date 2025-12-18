from paperAssignments.Assignments1_50.CA20.src import train, config


def test_train_runs_quickly(tmp_path):
    cfg = config.Config()
    cfg.epochs = 1
    cfg.batch_size = 16
    cfg.obs_dim = 4
    cfg.action_dim = 1
    res = train.train(cfg)
    assert "history" in res
    assert len(res["history"]) == cfg.epochs
    assert "checkpoint" in res












