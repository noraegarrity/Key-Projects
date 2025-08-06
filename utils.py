import os
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from translation import translate_zh_to_en_with_segmentation
from summarization import summarize_chinese_text
from evaluation import evaluate_summary


def read_input_text(input_path):
    """Reads the input text from a file."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            chinese_text = f.read()
        if not chinese_text.strip():
            raise ValueError(f"Input file '{input_path}' is empty or contains only whitespace.")
        return chinese_text
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found at {input_path}")


def handle_output(input_path, output_path, output_text):
    """Handles writing the output to the appropriate location."""

    if os.path.isdir(output_path):
        output_file_name = os.path.splitext(os.path.basename(input_path))[0] + "_output.txt"
        output_file_path = os.path.join(output_path, output_file_name)
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Results for '{os.path.basename(input_path)}' written to: {output_file_path}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
    elif os.path.isfile(output_path) or not os.path.exists(output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Results for '{os.path.basename(input_path)}' written to: {output_path}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
    else:
        print(f"Error: Invalid output path '{output_path}'. It must be an existing file or a directory.")


def process_with_evaluation(output_text, english_summary, reference_path):
    """Evaluates the summary if a reference is provided."""
    try:
        with open(reference_path, "r", encoding="utf-8") as ref_file:
            reference_summary = ref_file.read()
        evaluation_metrics = evaluate_summary(reference_summary, english_summary)
        output_text += "\n--- Automatic Evaluation Metrics ---\n\n"
        output_text += "ROUGE\n"
        output_text += (
            f"    ROUGE-1 F1: {evaluation_metrics['rouge1_fmeasure']:.4f},\n"
            f"    ROUGE-L F1: {evaluation_metrics['rougeL_fmeasure']:.4f}\n\n"
            # f"    BLEU: {evaluation_metrics['bleu']:.4f}\n"  # Preserving the commented-out line
        )
        output_text += "BERTScore\n"
        output_text += (
            f"    BERT Precision: {evaluation_metrics['bert_precision']:.4f},\n"
            f"    BERT Recall: {evaluation_metrics['bert_recall']:.4f},\n"
            f"    BERT F1: {evaluation_metrics['bert_f1']:.4f}\n"
        )
    except FileNotFoundError:
        print(f"Warning: Reference file not found at {reference_path}, skipping evaluation.")
    except Exception as e:
        print(f"Warning: Error during evaluation: {e}, skipping.")
    return output_text

def process_with_evaluation_dataset(output_text, english_summary, reference_summary):
    """Evaluates the summary if a reference is provided."""
    try:
        evaluation_metrics = evaluate_summary(reference_summary, english_summary)
        output_text += "\n--- Automatic Evaluation Metrics ---\n\n"
        output_text += "ROUGE\n"
        output_text += (
            f"    ROUGE-1 F1: {evaluation_metrics['rouge1_fmeasure']:.4f},\n"
            f"    ROUGE-L F1: {evaluation_metrics['rougeL_fmeasure']:.4f}\n\n"
            # f"    BLEU: {evaluation_metrics['bleu']:.4f}\n"  # Preserving the commented-out line
        )
        output_text += "BERTScore\n"
        output_text += (
            f"    BERT Precision: {evaluation_metrics['bert_precision']:.4f},\n"
            f"    BERT Recall: {evaluation_metrics['bert_recall']:.4f},\n"
            f"    BERT F1: {evaluation_metrics['bert_f1']:.4f}\n"
        )
    except Exception as e:
        print(f"Warning: Error during evaluation: {e}, skipping.")
    return output_text

def process_directory_evaluation(output_text, english_summary, filename, reference_dir_path):
    """Handles evaluation when processing directories."""

    reference_file_path = os.path.join(reference_dir_path, os.path.splitext(filename)[0] + "_ref.txt")
    if os.path.exists(reference_file_path):
        output_text = process_with_evaluation(output_text, english_summary, reference_file_path)
    else:
        output_text += f"    Warning: No corresponding reference file found for {filename}, skipping evaluation.\n"
        print(f"    Warning: No corresponding reference file found for {filename}, skipping evaluation.")
    return output_text