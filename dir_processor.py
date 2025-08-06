import os
import time
import sys  # Import the sys module

# Add main_app to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from translation import translate_zh_to_en_with_segmentation
from summarization import summarize_chinese_text
# from summarization import summarize_english_text
from evaluation import evaluate_summary
from utils import read_input_text, handle_output, process_with_evaluation, process_directory_evaluation  # Moved utils


def handle_directory_processing(input_dir_path, output_path, reference_dir_path=None):
    """Handles directory processing, determining output type."""
    if os.path.isfile(output_path) or not os.path.exists(output_path):
        handle_directory_to_single_file(input_dir_path, output_path, reference_dir_path)
    elif os.path.isdir(output_path):
        handle_directory_to_multiple_files(input_dir_path, output_path, reference_dir_path)
    else:
        raise ValueError(f"Invalid output path: '{output_path}'. It must be an existing file or a directory.")


def handle_directory_to_single_file(input_dir_path, output_file_path, reference_dir_path=None):
    """Processes all text files in an input directory and writes to a single output file."""

    try:
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            for filename in os.listdir(input_dir_path):
                if filename.endswith(".txt"):
                    input_file_path = os.path.join(input_dir_path, filename)
                    print(f"\nProcessing file: {input_file_path}")
                    try:
                        chinese_text = read_input_text(input_file_path)
                        output_text, english_summary = process_text(chinese_text)

                        if reference_dir_path:
                            output_text = process_directory_evaluation(output_text, english_summary, filename, reference_dir_path)

                        outfile.write(output_text)
                        print(f"Results for {filename} appended to {output_file_path}")

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        print(f"\nAll results written to: {output_file_path}")
    except Exception as e:
        print(f"Error writing to output file: {e}")


def handle_directory_to_multiple_files(input_dir_path, output_dir_path, reference_dir_path=None):
    """Processes all text files in an input directory and writes to separate output files."""

    for filename in os.listdir(input_dir_path):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_dir_path, filename)
            output_filename = os.path.splitext(filename)[0] + "_output.txt"
            output_file_path = os.path.join(output_dir_path, output_filename)
            print(f"\nProcessing file: {input_file_path}")
            try:
                chinese_text = read_input_text(input_file_path)
                output_text, english_summary = process_text(chinese_text)

                if reference_dir_path:
                    output_text = process_directory_evaluation(output_text, english_summary, filename, reference_dir_path)

                with open(output_file_path, "w", encoding="utf-8") as outfile:
                    outfile.write(output_text)
                print(f"Results for '{filename}' written to: {output_file_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def process_text(chinese_text):
    """Summarizes the given Chinese text and translates the summary to English."""

    total_start_time = time.time()
    chinese_summary = summarize_chinese_text(chinese_text)  # generate summary in Traditional Chinese
    summarize_time = time.time() - total_start_time

    english_translation, translation_time = translate_zh_to_en_with_segmentation(chinese_summary)  # translate to English
    total_processing_time = time.time() - total_start_time

    chinese_char_count = len(chinese_text)
    english_word_count = len(english_translation.split())  # Word counts for English Translation

    output_text = (
        f"English Translation of Chinese Summary:\n{english_translation}\n\n"
        f"Original Chinese Character Count: {chinese_char_count}\n"
        f"Summary English Word Count: {english_word_count}\n"
        f"Summarization Time: {summarize_time:.4f} seconds\n"
        f"Translation Time: {translation_time:.4f} seconds\n"
        f"Total Processing Time: {total_processing_time:.4f} seconds\n"
    )

    return output_text, english_translation  # return English summary for further process