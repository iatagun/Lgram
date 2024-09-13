gender_dict = {
    'Aaron': 'male',
    'Abigail': 'female',
    'Adam': 'male',
    'Alice': 'female',
    'Amanda': 'female',
    'Andrew': 'male',
    'Angela': 'female',
    'Ashley': 'female',
    'Benjamin': 'male',
    'Brianna': 'female',
    'Charles': 'male',
    'Charlotte': 'female',
    'Daniel': 'male',
    'David': 'male',
    'Emma': 'female',
    'Ethan': 'male',
    'Grace': 'female',
    'Hannah': 'female',
    'Isabella': 'female',
    'James': 'male',
    'Jessica': 'female',
    'John': 'male',
    'Julia': 'female',
    'Liam': 'male',
    'Linda': 'female',
    'Michael': 'male',
    'Mia': 'female',
    'Oliver': 'male',
    'Olivia': 'female',
    'Robert': 'male',
    'Samantha': 'female',
    'Sarah': 'female',
    'Sophia': 'female',
    'William': 'male',
    # Daha fazla isim eklenebilir
}

def get_gender(entities):
    genders = {}
    for entity in entities:
        if entity in gender_dict:
            genders[entity] = gender_dict[entity]
    return genders