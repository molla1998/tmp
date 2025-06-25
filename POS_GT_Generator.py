import random
import json

# Word banks
nouns = ["phone", "laptop", "charger", "headphones", "tablet", "watch", "camera", "speaker", "mic", "router", "printer", "monitor", "keyboard", "mouse", "projector", "powerbank", "case", "screen", "earbuds", "gamepad"]
adjs = ["cheap", "fast", "wireless", "affordable", "powerful", "compatible", "new", "slim", "durable", "portable", "lightweight", "stylish", "reliable", "advanced", "smart"]
nums = ["10000", "500", "65w", "256gb", "2tb", "50", "100", "2000", "300", "150"]
adps = ["with", "for", "under", "above", "beside", "without"]

# 50 sentence templates
templates = [
    "Give me {adj1} {noun1} with {adj2} {noun2}.",
    "I want {noun1} for {adj2} {noun2}.",
    "Find {adj1} {noun1} under {num}.",
    "Looking for {adj1} {noun1} with {adj2} {noun2}.",
    "Buy {adj1} {noun1} with {adj2} {noun2}.",
    "Get {noun1} for {adj2} {noun2}.",
    "Search {adj1} {noun1} under {num}.",
    "Order {adj1} {noun1} with {adj2} {noun2}.",
    "Cheap {noun1} for {adj2} {noun2}.",
    "Looking for {adj1} {noun1} above {num}.",
    "I need {adj1} {noun1} beside {adj2} {noun2}.",
    "Get {noun1} without {adj2} {noun2}.",
    "Find {adj1} {noun1} above {num}.",
    "Order {adj1} {noun1} without {adj2} {noun2}.",
    "New {noun1} with {adj2} {noun2}.",
    "Affordable {noun1} for {adj2} {noun2}.",
    "Looking for {adj1} {noun1} with {adj2} {noun2}.",
    "Buy {adj1} {noun1} for {adj2} {noun2}.",
    "Search for {adj1} {noun1} under {num}.",
    "Get {adj1} {noun1} above {num}.",
    "Order {adj1} {noun1} beside {adj2} {noun2}.",
    "Find {adj1} {noun1} without {adj2} {noun2}.",
    "Need {adj1} {noun1} with {adj2} {noun2}.",
    "Looking for {adj1} {noun1} for {adj2} {noun2}.",
    "Cheap {adj1} {noun1} under {num}.",
    "Buy {adj1} {noun1} with {adj2} {noun2}.",
    "Order {adj1} {noun1} for {adj2} {noun2}.",
    "I want {adj1} {noun1} with {adj2} {noun2}.",
    "Need {adj1} {noun1} under {num}.",
    "Searching for {adj1} {noun1} above {num}.",
    "Purchase {adj1} {noun1} without {adj2} {noun2}.",
    "Get {adj1} {noun1} with {adj2} {noun2}.",
    "Find {adj1} {noun1} beside {adj2} {noun2}.",
    "Affordable {adj1} {noun1} above {num}.",
    "New {adj1} {noun1} for {adj2} {noun2}.",
    "Order {adj1} {noun1} under {num}.",
    "Need {adj1} {noun1} with {adj2} {noun2}.",
    "Searching {adj1} {noun1} for {adj2} {noun2}.",
    "Get {adj1} {noun1} under {num}.",
    "Looking for {adj1} {noun1} without {adj2} {noun2}.",
    "Purchase {adj1} {noun1} for {adj2} {noun2}.",
    "I want {adj1} {noun1} above {num}.",
    "Find {adj1} {noun1} beside {adj2} {noun2}.",
    "Buy {adj1} {noun1} without {adj2} {noun2}.",
    "Affordable {adj1} {noun1} with {adj2} {noun2}.",
    "Order {adj1} {noun1} for {adj2} {noun2}.",
    "Looking for {adj1} {noun1} beside {adj2} {noun2}.",
    "Purchase {adj1} {noun1} above {num}.",
    "Need {adj1} {noun1} without {adj2} {noun2}.",
    "Cheap {adj1} {noun1} for {adj2} {noun2}."
]

test_dataset = []

for template in templates:
    adj1 = random.choice(adjs)
    adj2 = random.choice(adjs)
    noun1 = random.choice(nouns)
    noun2 = random.choice(nouns)
    num = random.choice(nums)

    sentence = template.format(adj1=adj1, adj2=adj2, noun1=noun1, noun2=noun2, num=num)

    # Basic rule-based expected output logic
    expected = {}

    if " with " in template and " for " not in template:
        expected = {
            "Primary main noun": noun1,
            "Primary prev nouns": "",
            "Adj ref Primary noun": adj1,
            "ADP ref Primary noun": "",
            "Verb ref Primary noun": "",
            "Secondary main noun": noun2,
            "Secondary next nouns": "",
            "Adj ref Secondary noun": adj2,
            "ADP ref Secondary noun": "",
            "Verb ref Secondary noun": ""
        }
    elif " for " in template:
        expected = {
            "Primary main noun": noun2,
            "Primary prev nouns": "",
            "Adj ref Primary noun": adj2,
            "ADP ref Primary noun": "",
            "Verb ref Primary noun": "",
            "Secondary main noun": noun1,
            "Secondary next nouns": "",
            "Adj ref Secondary noun": adj1,
            "ADP ref Secondary noun": "",
            "Verb ref Secondary noun": ""
        }
    elif " under " in template or " above " in template:
        adp = "under" if " under " in template else "above"
        expected = {
            "Primary main noun": noun1,
            "Primary prev nouns": "",
            "Adj ref Primary noun": adj1,
            "ADP ref Primary noun": adp,
            "Verb ref Primary noun": "",
            "Secondary main noun": num,
            "Secondary next nouns": "",
            "Adj ref Secondary noun": "",
            "ADP ref Secondary noun": "",
            "Verb ref Secondary noun": ""
        }
    else:
        expected = {
            "Primary main noun": noun1,
            "Primary prev nouns": "",
            "Adj ref Primary noun": adj1,
            "ADP ref Primary noun": "",
            "Verb ref Primary noun": "",
            "Secondary main noun": noun2,
            "Secondary next nouns": "",
            "Adj ref Secondary noun": adj2,
            "ADP ref Secondary noun": "",
            "Verb ref Secondary noun": ""
        }

    test_dataset.append({
        "text": sentence,
        "expected": expected
    })

# Save as JSON
with open("generated_ground_truth_dataset.json", "w") as f:
    json.dump(test_dataset, f, indent=4)

print("Generated 50 ground truth examples saved to 'generated_ground_truth_dataset.json'")
