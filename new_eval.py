def eval_main_product_with_adj(gold, pred):
    """
    gold: [text, start, end, adj_list, adp_list, label, is_main_product]
    pred: [text, start, end, label, adj_list, adp_list, is_main_product]

    Evaluates:
        - product_name/accessory entity (overlap matching)
        - adj list (set TP, FP, FN)
        - is_main_product TP/FP/FN
    """

    from collections import defaultdict

    VALID_PRODUCT_LABELS = {"product_name", "accessory"}

    entity_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    adj_metrics = {"tp": 0, "fp": 0, "fn": 0}
    main_metrics = {"tp": 0, "fp": 0, "fn": 0}

    used_pred = set()

    # ----------- MATCH MAIN PRODUCT + ENTITY MATCHES ---------------- #

    for gi, g in enumerate(gold):
        g_text, g_s, g_e, g_adj, _, g_label, g_main = g
        
        # Only evaluate entities for product_name/accessory
        if g_label not in VALID_PRODUCT_LABELS:
            continue

        g_adj_set = set([x.strip() for x in g_adj.split(",") if x.strip()])
        
        matched_pred = None

        for pi, p in enumerate(pred):
            if pi in used_pred:
                continue

            p_text, p_s, p_e, p_label, p_adj, _, p_main = p

            # Label must be valid product-related label
            if p_label not in VALID_PRODUCT_LABELS:
                continue

            # Overlap rule
            overlap = not (p_e < g_s or p_s > g_e)
            if not overlap:
                continue

            # ENTITIES MATCH
            matched_pred = (pi, p)
            used_pred.add(pi)

            # ------------------- ADJ METRICS ---------------------------#
            p_adj_set = set([x.strip() for x in p_adj.split(",") if x.strip()])

            if not g_adj_set and not p_adj_set:
                # both blank → full adj TP
                pass
            else:
                # TP adj = intersection
                adj_metrics["tp"] += len(g_adj_set.intersection(p_adj_set))
                # FN adj = gold missing in pred
                adj_metrics["fn"] += len(g_adj_set - p_adj_set)
                # FP adj = pred extra adj
                adj_metrics["fp"] += len(p_adj_set - g_adj_set)

            # ------------------- ENTITY TP -----------------------------#
            entity_metrics[g_label]["tp"] += 1

            # ------------------- MAIN PRODUCT --------------------------#
            if g_main is True:
                if p_main is True:
                    main_metrics["tp"] += 1
                else:
                    main_metrics["fn"] += 1

            break

        # If NO pred matched this gold entity
        if matched_pred is None:
            entity_metrics[g_label]["fn"] += 1

            # main product FN
            if g_main is True:
                main_metrics["fn"] += 1

    # ---------- COUNT ENTITY FP + MAIN PRODUCT FP ---------------- #

    for pi, p in enumerate(pred):
        if pi in used_pred:
            continue

        _, _, _, p_label, p_adj, _, p_main = p

        if p_label in VALID_PRODUCT_LABELS:
            entity_metrics[p_label]["fp"] += 1

        # FP main product: predicted main but doesn’t match any gold
        if p_main is True:
            main_metrics["fp"] += 1

        # FP adj (if pred has adj but no match)
        p_adj_set = set([x.strip() for x in p_adj.split(",") if x.strip()])
        adj_metrics["fp"] += len(p_adj_set)

    # -------- RETURN COMPLETE RESULTS ---------- #

    return {
        "entity": entity_metrics,
        "adj": adj_metrics,
        "main_product": main_metrics
    }
