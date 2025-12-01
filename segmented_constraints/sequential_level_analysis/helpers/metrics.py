# helpers/metrics.py
def parse_target(relation: str, target_str: str):
    r = relation.lower()
    if r == "range":
        lo_s, hi_s = target_str.split("-")
        return int(lo_s), int(hi_s)
    return int(target_str)

def hard_metric(relation: str, actual: int, target):
    if relation == "range":
        lo, hi = target
        return 1 if lo <= actual <= hi else 0
    elif relation == "gte":
        return 1 if actual >= target else 0
    elif relation == "lte":
        return 1 if actual <= target else 0
    elif relation == "approx":
        lo = round(0.95 * target)
        hi = round(1.05 * target)
        return 1 if lo <= actual <= hi else 0
    else:
        return 0

