import os
import time
import sys  # Import the sys module

# Add main_app to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from translation import translate_zh_to_en_with_segmentation
from summarization import summarize_chinese_text
from evaluation import evaluate_summary
from utils import read_input_text, handle_output, process_with_evaluation  # Moved common utils


def handle_single_file(input_file_path, output_path, reference_path=None):
    """Handles the processing of a single input file."""
    try:
        chinese_text = read_input_text(input_file_path)
        output_text, english_summary = process_text(chinese_text)

        if reference_path:
            output_text = process_with_evaluation(output_text, english_summary, reference_path)

        handle_output(input_file_path, output_path, output_text)

    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")

def process_text(chinese_text):
    """Summarize the given Chinese text and translate the summary to English."""

    total_start_time = time.time()
    chinese_summary = summarize_chinese_text(chinese_text)
    summarize_time = time.time() - total_start_time
    english_translation, translation_time = translate_zh_to_en_with_segmentation(chinese_summary)
    total_processing_time = time.time() - total_start_time

    chinese_char_count = len(chinese_text)
    english_word_count = len(english_translation.split())

    output_text = (
        f"English Translation of Chinese Summary:\n{english_translation}\n\n"
        f"Original Chinese Character Count: {chinese_char_count}\n"
        f"Summary English Word Count: {english_word_count}\n"
        f"Summarization Time: {summarize_time:.4f} seconds\n"
        f"Translation Time: {translation_time:.4f} seconds\n"
        f"Total Processing Time: {total_processing_time:.4f} seconds\n"
    )

    return output_text, english_translation