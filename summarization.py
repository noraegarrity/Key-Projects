from transformers import pipeline

def summarize_chinese_text(text, model_name="facebook/mbart-large-50", max_length=200, min_length=50):
    """
    Generates a Traditional Chinese summary of the given text.

    Args:
        text (str): The Traditional Chinese text to summarize.
        model_name (str, optional): The name of the summarization model to use.
        max_length (int, optional): The maximum length of the summary.
        min_length (int, optional): The minimum length of the summary.

    Returns:
        str: The generated Traditional Chinese summary.
    """

    summarizer = pipeline("summarization", model=model_name, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
    summary_result = summarizer(text)[0]['summary_text']
    return summary_result