# Summarization: Summarize-then-Translate

This tool summarizes Traditional Chinese text files and translates the summaries to English. Then, evaluates the summaries if a reference is provided.
> While some parts of this file are identical or only minimally different from the baseline system, we have included all required files in the D3 folder to ensure clear separation and prevent confusion during execution. This setup allows users to test the complete pipeline conveniently from a single location.


## Pre-requisites

> Please note that as of now, you should have followed **steps 1 to 6** listed on `README.md`.

## Description of Files

* `run.py`: The main entry point of the program. It uses `argparse` to handle command-line arguments and calls functions from `file_processor.py` or `dir_processor.py` depending on whether the input is a file or a directory.

* `file_processor.py`: Handles the processing of a single input file. It's responsible for reading the Chinese text, generating a summary, translating it to English, and handling output. If a reference summary is provided, it also performs evaluation.

* `dir_processor.py`: Handles the processing of input directories. It can write the processing results of multiple files to a single output file or generate separate output files for each input file.

* `summarization.py`: Contains the function `summarize_chinese_text` to generate Chinese summaries, using the Transformers library's `pipeline`.

* `translation.py`: Contains the function `translate_zh_to_en_with_segmentation` to translate Chinese summaries to English, also using Transformers. It supports segmented translation for long texts.

* `evaluation.py`: Contains the function `evaluate_summary` to evaluate the quality of summaries, calculating ROUGE and BERTScore metrics.

* `utils.py`: Contains utility functions, such as reading input text, handling output, and performing summary evaluation.


> The test_data folder contains example input and output files for users' reference.


## How to Run `run.py`

1.  **Prepare Input Files:** Store the Traditional Chinese text you want to process as `.txt` files. This is the text you want the program to summarize.
2.  **Execute `run.py`:** Use Python to run `run.py` from the command line, providing the necessary arguments.

    ```bash
    python run.py -i <input_path> -o <output_path> [-r <reference_path>]
    ```

    * `<input_path>`: (Required) The path to the input file (`.txt`) or directory. If the input is a directory, all `.txt` files in that directory will be processed. This is the Chinese text to be summarized.
    * `<output_path>`: (Optional) The path to the output file or directory.
        * If not provided, defaults to `out.txt`.
        * If a file path is provided, all processing results will be written to the same file.
        * If a directory path is provided, a corresponding `*_output.txt` file will be generated for each input file.
    * `<reference_path>`: (Optional) The path to the reference English summary file/directory. If provided, evaluation metrics will be calculated. This is the "ground truth" or expected summary, typically created by a human, and used to evaluate the quality of the automatically generated summary.
        * If it's a file, it's used to evaluate the summary of a single input file.
        * If it's a directory, it's assumed that each `*_ref.txt` file in the directory corresponds to an input `.txt` file with the same name (e.g., `input.txt` corresponds to `input_ref.txt`). **These reference files should contain the expected, high-quality English summaries against which the program's output will be compared.**

## Examples

* **Process a single file and output to a file:**

    ```bash
    python run.py -i input.txt -o output.txt
    ```

* **Process a single file and output to a directory:**

    ```bash
    python run.py -i input.txt -o output_dir/
    ```

* **Process a directory and output to a single file:**

    ```bash
    python run.py -i input_dir/ -o output.txt
    ```

* **Process a directory and output to multiple files:**

    ```bash
    python run.py -i input_dir/ -o output_dir/
    ```

* **Process a single file and perform evaluation:**

    ```bash
    python run.py -i input.txt -o output.txt -r reference.txt
    ```

* **Process a directory and perform evaluation:**

    ```bash
    python run.py -i input_dir/ -o output_dir/ -r reference_dir/
    ```

## Output Format

The program's output includes the following information:

* English translation of the Chinese summary
* Original Chinese text character count
* English summary word count
* Summarization time
* Translation time
* Total processing time
* Automatic evaluation metrics (if a reference summary is provided):
    * ROUGE-1 F1
    * ROUGE-L F1
    * BERTScore Precision
    * BERTScore Recall
    * BERTScore F1

```text
English Translation of Chinese Summary:
During a memorial ceremony at the Church of the Westminster, guests attended, in addition to literature, films, dramas, and the media, was one of the largest gatherings of the late generations of Dickens. Prince Charles was to offer a wreath on Charles Dickens’ tombstone.

Original Chinese Character Count: 566
Summary English Word Count: 44
Summarization Time: 15.8311 seconds
Translation Time: 0.8111 seconds
Total Processing Time: 17.3464 seconds

--- Automatic Evaluation Metrics ---

ROUGE
    ROUGE-1 F1: 0.4054,
    ROUGE-L F1: 0.2703

BERTScore
    BERT Precision: 0.8353,
    BERT Recall: 0.8490,
    BERT F1: 0.8421
