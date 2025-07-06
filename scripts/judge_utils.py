from typing import Any, Dict


def merge_judge_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Combine provider results into a single mapping with a ``flagged`` list.

    If ``results`` already contains a top-level ``"flagged"`` key, it is
    returned unchanged. Otherwise, each value in ``results`` is inspected for a
    ``"flagged"`` list which will be concatenated in the returned dictionary.
    """
    if not isinstance(results, dict):
        return {}
    if "flagged" in results:
        return results

    combined = {"flagged": []}
    for value in results.values():
        if isinstance(value, dict):
            flagged = value.get("flagged")
            if isinstance(flagged, list):
                combined["flagged"].extend(flagged)
    return combined
