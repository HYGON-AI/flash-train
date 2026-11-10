"""Comparable workload identities for baseline pairing and report grouping."""

import json


def workload_key(row):
    return (row.get("op"), row.get("precision"), tuple(row["shape"]),
            json.dumps(row.get("workload", {}), sort_keys=True, separators=(",", ":")))


def row_key(row):
    return (row.get("tier", "convenience-python"), *workload_key(row),
            row.get("impl"), row.get("index"))


def workload_label(row):
    if row.get("workload_label"):
        return row["workload_label"]
    workload = row.get("workload")
    if not workload:
        return ""
    return " / ".join(f"{name}={value}" for name, value in sorted(workload.items()))
