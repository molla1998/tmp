def flexible_span_overlap(gt, pred, tolerance=2, min_overlap_ratio=0.3):
    """
    gt  = [text, start, end, adj, adp, is_main]
    pred = [text, start, end, adj, adp, is_main, label]

    New rule added:
    - If gt_text is inside pred_text OR pred_text is inside gt_text → consider TP
      (helps cases like 'phone' vs 'phon', 'cab' vs 'cabs', etc.)
    """

    g_text = str(gt[0]).lower().strip()
    p_text = str(pred[0]).lower().strip()

    # 1 ) NEW RULE: substring containment match
    if g_text and p_text:
        if g_text in p_text or p_text in g_text:
            return True

    # 2 ) Normal span integer checks
    try:
        gs, ge = int(gt[1]), int(gt[2])
        ps, pe = int(pred[1]), int(pred[2])
    except:
        return False

    # 3 ) Soft boundary tolerance
    if abs(gs - ps) <= tolerance and abs(ge - pe) <= tolerance:
        return True

    # 4 ) Overlap ratio check
    overlap = max(0, min(ge, pe) - max(gs, ps))
    gt_len = max(1, ge - gs)
    pred_len = max(1, pe - ps)

    ratio = overlap / min(gt_len, pred_len)

    return ratio >= min_overlap_ratio
