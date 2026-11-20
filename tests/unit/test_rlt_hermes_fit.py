from datagoblin.training_gym.rlt_hermes_fit import menu, summarize


def test_projection_menu_checks_only_candidates_it_can_commit():
    class Campsite:
        def __init__(self):
            self.checks = []
            self.attempts = []

        def passes(self, task, candidate):
            self.checks.append(candidate)
            return candidate == "verified"

        def repair(self, task, public_constraints, grid, operator):
            self.attempts.append(operator)
            return "verified" if operator == "dual_repair" else None

    verifier = Campsite()
    assert menu(verifier, None, None, "verified") == ("original", 1, 0)
    assert verifier.attempts == []
    assert menu(verifier, None, None, "broken") == ("dual_repair", 2, 2)
    assert verifier.checks == ["verified", "broken", "verified"]


def test_stopping_counts_early_pass_lost_at_final_cycle():
    parsed = {"parsed": True, "rectangular": True, "menu_verifier_calls": 3, "projection_attempts": 2}
    rows = [
        {**parsed, "menu_operator": "dual_repair", "cycle_passes": [True, True, True]},
        {**parsed, "menu_operator": None, "cycle_passes": [True, False, False]},
        {**parsed, "menu_operator": None, "cycle_passes": [False, False, True]},
        {**parsed, "menu_operator": None, "cycle_passes": [False, False, False]},
    ]
    result = summarize(rows, 3, 3)
    assert result["fixed_solved"] == 2
    assert result["stop_solved"] == 3
    assert result["first_valid_cycle_on_open_cases"] == [1, 0, 1]
    assert result["fixed_policy_verifier_calls"] == 15
    assert result["stop_policy_verifier_calls"] == 19
    assert result["fixed_policy_block_calls"] == 36
    assert result["stop_policy_block_calls"] == 28
