import importlib

planner = importlib.import_module("planner")


def test_package_exports():
    # ensure convenient exports exist
    assert hasattr(planner, "CheckpointBuffer")
    assert hasattr(planner, "simulate_branches")
    assert hasattr(planner, "should_trigger")
    assert hasattr(planner, "TriggerConfig")
    assert hasattr(planner, "branch_to_action_distribution")
    assert hasattr(planner, "select_branch_action")
