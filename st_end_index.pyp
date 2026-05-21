def post_process(sample):

    entities = []

    current = None

    tokens = sample["tokens"]

    for tok in tokens:

        ner = tok["ner"]

        ####################################################
        # OUTSIDE
        ####################################################
        if ner == "O":

            if current:
                entities.append(current)
                current = None

            continue

        ####################################################
        # SPLIT BIO
        ####################################################
        prefix, label = ner.split("-", 1)

        ####################################################
        # BEGIN ENTITY
        ####################################################
        if prefix == "B":

            if current:
                entities.append(current)

            current = {
                "entity": label,
                "text": tok["token"],
                "start": tok["start"],
                "end": tok["end"]
            }

            ################################################
            # main_product only for required labels
            ################################################
            if (
                label in {
                    "PRODUCT_NAME",
                    "ACCESSORY"
                }
                and "is_main_product" in tok
            ):

                current["is_main_product"] = (
                    tok["is_main_product"]
                )

        ####################################################
        # INSIDE ENTITY
        ####################################################
        elif prefix == "I" and current:

            token_text = tok["token"]

            ################################################
            # HANDLE WORDPIECE TOKENS
            ################################################
            if token_text.startswith("##"):

                current["text"] += (
                    token_text[2:]
                )

            else:

                current["text"] += (
                    " " + token_text
                )

            ################################################
            # UPDATE END POSITION
            ################################################
            current["end"] = tok["end"]

            ################################################
            # merge main_product
            ################################################
            if (
                label in {
                    "PRODUCT_NAME",
                    "ACCESSORY"
                }
                and "is_main_product" in tok
            ):

                current["is_main_product"] = (
                    current.get(
                        "is_main_product",
                        False
                    )
                    or tok["is_main_product"]
                )

    ########################################################
    # LAST ENTITY
    ########################################################
    if current:
        entities.append(current)

    return {
        "intent": sample["intent"],
        "entities": entities
    }
