import numpy as np

DEFAULT_THRESHOLD = 0.5


def find_revisit_events(agent_pos, threshold=DEFAULT_THRESHOLD):
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
