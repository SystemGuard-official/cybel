import spacy

def filter_stopwords(text, model="en_core_web_sm"):
    """
    Removes stop words from the input text using spaCy.

    Args:
        text (str): The input text to filter.
        model (str): The spaCy language model to use (default is "en_core_web_sm").

    Returns:
        list: A list of words from the text without stop words.

    Raises:
        ValueError: If the input text is not a string.
        OSError: If the specified spaCy model is not found.
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    try:
        nlp = spacy.load(model)
    except OSError as e:
        raise OSError(f"Error loading spaCy model '{model}'. Ensure it is installed.") from e

    # Process text
    doc = nlp(text)

    # Filter out stop words
    filtered_words = [token.text for token in doc if not token.is_stop]

    cleaned_output = " ".join(filtered_words)

    return cleaned_output


# Example usage
# if __name__ == "__main__":
#     try:
#         input_text = "This is an example showing off stop word filtration."
#         result = filter_stopwords(input_text)
#         print("Filtered Words:", result)
#     except Exception as e:
#         print(f"An error occurred: {e}")
