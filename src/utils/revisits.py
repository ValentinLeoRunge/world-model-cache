"""Find revisit events in a recorded trajectory: times the agent returned to a
place it had already left. See blueprints/WORLDMODEL.md (Evaluation) and
ACTION_PLAN.md step 1 ("find all revisit events, same place, N steps apart").
"""

import numpy as np

# "same place" radius, in agent_pos units. Calibrated from data/tokens/: the
# per-step displacement is a near-constant ~0.25 (fixed sub-step size), and the
# maze grid is 9x9 with ~1.0 units/cell. 0.5 sits between the two -- bigger
# than two ordinary steps (so gentle drift near the current spot doesn't count
# as a separate visit), smaller than one full cell (so it means "this spot",
# not "this room").
DEFAULT_THRESHOLD = 0.5


def find_revisit_events(agent_pos, threshold=DEFAULT_THRESHOLD):
    """agent_pos: (T,2) array-like. -> list of (t1, t2, N) with N = t2 - t1.

    For each t2, scans backward past the tail of the *current* visit (recent
    positions already within `threshold` of pos[t2] -- the agent hasn't left
    yet), then through the "away" gap, until it finds the most recent earlier
    time t1 also within `threshold`. That is a revisit: the agent left this
    spot and returned N = t2 - t1 steps later. Skipped if the agent never left
    (still the same visit) or never visited this spot before at all.
    """
    pos = np.asarray(agent_pos, dtype=np.float64)
    t_total = len(pos)
    events = []

    for t2 in range(1, t_total):
        dists = np.linalg.norm(pos[:t2] - pos[t2], axis=1)  # (t2,), times 0..t2-1
        close = dists < threshold

        rev_close = close[::-1]                 # walking backward: t2-1, t2-2, ..., 0
        away = ~rev_close
        if not away.any():
            continue                            # always close: hasn't left yet
        leave_at = int(np.argmax(away))          # steps back to the first "away" point

        still_close = rev_close[leave_at:]
        if not still_close.any():
            continue                            # left, but no earlier visit at all
        return_at = leave_at + int(np.argmax(still_close))

        t1 = t2 - 1 - return_at
        events.append((t1, t2, t2 - t1))

    return events
