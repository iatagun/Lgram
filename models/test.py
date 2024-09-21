from centering_model import CenteringModel

text = (
    "Alice was excited about her upcoming vacation. She had been planning it for months. "
    "Her friend Bob decided to join her. They both agreed that visiting Paris would be the highlight of the trip."
)

model = CenteringModel(text)
total_score = model.score_transitions()

# Anlaşılır bir formatta çıktı için
for idx, score in enumerate(model.scores, start=1):
    print(f"--- Transition {idx} ---")
    print(f"Current Sentence: {score['current_sentences']}")
    print(f"Next Sentence: {score['next_sentences']}")
    print(f"Transition Type: {score['transition']}")
    print(f"Score: {score['score']}")
    print()  # Boş satır ile ayrım yap

# Toplam skor
print(f"Total Transition Score: {total_score}")

