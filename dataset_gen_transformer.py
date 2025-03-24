import random
import csv

def generate_combinations(n):
    PDT_GEN = ["phone", "galaxy phone", "laptop", "charger"]
    PDT_MODEL = ["s24", "galaxy s24"]
    VERB = ["working", "cooling"]
    SPEC = ["ram", "price"]
    
    templates = [
        "{PDT_GEN} not {VERB}",
        "{PDT_GEN} for {PDT_MODEL}",
        "{PDT_GEN} installation",
        "{PDT_MODEL} features & specs",
        "{PDT_GEN} by {SPEC}"
    ]
    
    combinations = []
    for _ in range(n):
        template = random.choice(templates)
        filled_template = template.format(
            PDT_GEN=random.choice(PDT_GEN),
            PDT_MODEL=random.choice(PDT_MODEL),
            VERB=random.choice(VERB),
            SPEC=random.choice(SPEC)
        )
        combinations.append(filled_template)
    
    return combinations

def save_to_csv(data, filename="generated_combinations.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["NLQ"])  # Adding the header
        for row in data:
            writer.writerow([row])

if __name__ == "__main__":
    N = int(input("Enter the number of combinations: "))
    results = generate_combinations(N)
    save_to_csv(results)
    print(f"Results saved to 'generated_combinations.csv' successfully!")
