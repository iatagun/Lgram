from centering_model import CenteringModel

text = (
    "Alice was excited about her upcoming vacation. She had been planning it for months. "
    "Her friend Bob decided to join her. They both agreed that visiting Paris would be the highlight of the trip. "
    "Bob had always dreamed of seeing the Eiffel Tower. Later, they discussed what to do in the city."
)

model = CenteringModel(text)
total_score = model.score_transitions()

for score in model.scores:
    print(f"Transition from: {score['current_sentences']} to {score['next_sentences']}")
    print(f"Transition Type: {score['transition']}, Score: {score['score']}")

print(f"Total Transition Score: {total_score}")
