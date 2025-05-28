import random

subjects = [
    "elizabeth", "darcy", "miss bingley", "mrs hurst", "mr hurst", 
    "eliza bennet", "bingley", "sister", "lady", "woman", 
    "assistant", "physician", "housekeeper", "mr jones"
]

actions = [
    "said", "replied", "cried", "added", "observed", "joined", "recommended", 
    "called", "left", "described", "knew", "undervalue", "doubt", 
    "employ", "succeed", "complain", "propose", "settled", "sent"
]

objects = [
    "accomplishments", "music", "singing", "drawing", "dancing", 
    "languages", "voice", "address", "expressions", "capacity", 
    "taste", "application", "elegance", "mind", "reading", 
    "device", "art", "relief", "attention"
]

places = [
    "room", "town", "country", "house", "morning", "supper"
]



def generate_sentence():
    subj = random.choice(subjects)
    verb = random.choice(actions)
    obj = random.choice(objects)
    loc = random.choice(places)
    return f"{subj} {verb} {obj} {loc}."

with open("daily_life_dataset.txt", "w", encoding="utf-8") as f:
    for _ in range(10000):  # 10K satır üret
        f.write(generate_sentence() + "\n")
