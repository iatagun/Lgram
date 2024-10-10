from analyze_transitions import analyze_transitions 
def load_text_data_in_chunks(file_path, chunk_size=1000):
    """
    Reads the file in chunks of specified size.
    
    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Number of characters to read per chunk.
    
    Yields:
        str: The next chunk of text.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)  # Read chunk of text
            if not chunk:
                break
            yield chunk

# Path to your large text file
file_path = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_data.txt'

# Iterate over the file in chunks and process each one
for chunk in load_text_data_in_chunks(file_path, chunk_size=5000):  # Adjust chunk size as needed
    transition_df = analyze_transitions(chunk)  # Analyze transitions in each chunk
    
    # You can process each chunk's transition_df here, for example:
    print(transition_df)  # Output the analysis of the current chunk (can be saved or aggregated)

    # Optionally save or aggregate results for each chunk
    # transition_df.to_csv('chunk_transitions.csv', mode='a', header=False)  # Append results to a file

print("Processing complete.")
