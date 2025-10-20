# helpers/metrics.py
def parse_target(relation: str, target_str: str):
    r = relation.lower()
    if r == "range":
        lo_s, hi_s = target_str.split("-")
        return int(lo_s), int(hi_s)
    return int(target_str)

def hard_metric(relation: str, actual: int, target):
    r = relation.lower()
    if r == "range":
        lo, hi = target
        return lo <= actual <= hi
    if r == "gte":
        return actual >= target
    if r == "lte":
        return actual <= target
    if r == "approx":
        t = int(target)
        lo = round(0.9 * t)
        hi = round(1.1 * t)
        return lo <= actual <= hi
    return False
