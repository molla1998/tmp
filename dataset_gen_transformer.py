import random

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

if __name__ == "__main__":
    N = int(input("Enter the number of combinations: "))
    results = generate_combinations(N)
    for result in results:
        print(result)
