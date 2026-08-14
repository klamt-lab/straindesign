

def test_final_status_does_not_promote_a_failed_enumeration():
    """A solver that failed mid-enumeration leaves a truncated pool. Reporting that as OPTIMAL
    presents it as complete -- which is what a dropped licence produced: 38 of 438 designs,
    status optimal. Only exhaustion and the time limit may be rewritten."""
    from straindesign.compute_strain_designs import _final_status
    from straindesign.names import OPTIMAL, INFEASIBLE, TIME_LIMIT, TIME_LIMIT_W_SOL, ERROR

    assert _final_status(INFEASIBLE, True) == OPTIMAL
    assert _final_status(TIME_LIMIT, True) == TIME_LIMIT_W_SOL
    assert _final_status(ERROR, True) == ERROR
    assert _final_status(ERROR, False) == ERROR
    assert _final_status(INFEASIBLE, False) == INFEASIBLE
    assert _final_status(OPTIMAL, True) == OPTIMAL
