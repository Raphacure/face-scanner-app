from app.core.grouping import GROUP_MAP

def group_metrics(flat_results):
    grouped = {}

    for group, keys in GROUP_MAP.items():
        grouped[group] = {}

        for key in keys:
            # Normal metrics
            if key in flat_results:
                grouped[group][key] = flat_results[key]

        # remove empty groups
        if not grouped[group]:
            grouped.pop(group)

    # ✅ Add Skin Group Separately (nested object)
    if "skin" in flat_results:
        grouped["skin"] = flat_results["skin"]

    return grouped
