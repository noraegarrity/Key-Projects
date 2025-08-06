import argparse
import os
from transformers import MarianTokenizer, MarianMTModel, pipeline
import time

def translate_zh_to_en_with_segmentation(text, model_name='Helsinki-NLP/opus-mt-zh-en', max_tokens=512):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=False)
    input_len = input_ids.size(1)
    translated_chunks = []
    total_translation_time = 0

    if input_len > max_tokens:
        print(f"Input text exceeds the translation model ({model_name})'s maximum token limit ({max_tokens}), performing segmented translation.")
        num_chunks = (input_len + max_tokens - 1) // max_tokens
        for i in range(num_chunks):
            start_index = i * max_tokens
            end_index = min((i + 1) * max_tokens, input_len)
            chunk_input_ids = input_ids[:, start_index:end_index]

            chunk_start_time = time.time()
            outputs = model.generate(chunk_input_ids)
            chunk_end_time = time.time()
            chunk_translation_time = chunk_end_time - chunk_start_time
            total_translation_time += chunk_translation_time

            translated_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_chunks.append(translated_chunk)
            print(f"Translated segment {i+1}/{num_chunks} in {chunk_translation_time:.4f} seconds")
        translated_text = " ".join(translated_chunks)
    else:
        start_time = time.time()
        outputs = model.generate(input_ids)
        end_time = time.time()
        total_translation_time = end_time - start_time
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Translation completed in {total_translation_time:.4f} seconds")

    return translated_text, total_translation_time

def summarize_english_text(text, model_name="facebook/bart-large-cnn", max_length=150, min_length=30):
    summarizer = pipeline("summarization", model=model_name, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
    summary_result = summarizer(text)[0]['summary_text']
    return summary_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translates Chinese text and generates an English summary.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input Traditional Chinese text file or directory.")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file or directory for the English summary and statistics (optional, defaults to ./out.txt).")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if output_path is None:
        output_path = "out.txt"
        print(f"Output path not specified, defaulting to: {output_path}")

    if os.path.isfile(input_path):
        # Input is a file
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                chinese_text = f.read()
            if not chinese_text.strip():
                raise ValueError(f"Input file '{input_path}' is empty or contains only whitespace.")

            total_start_time = time.time()
            translate_start_time = time.time()
            english_translation, translation_time = translate_zh_to_en_with_segmentation(chinese_text)
            translate_end_time = time.time()
            overall_translation_time = translate_end_time - translate_start_time

            summarize_start_time = time.time()
            english_summary = summarize_english_text(english_translation)
            summarize_end_time = time.time()
            summarization_time = summarize_end_time - summarize_start_time

            total_end_time = time.time()
            total_processing_time = total_end_time - total_start_time

            chinese_char_count = len(chinese_text)
            english_word_count = len(english_summary.split())

            output_text = (
                f"English Summary:\n{english_summary}\n\n"
                f"Original Chinese Character Count: {chinese_char_count}\n"
                f"Summary English Word Count: {english_word_count}\n"
                f"Translation Time: {translation_time:.4f} seconds\n"
                f"Summarization Time: {summarization_time:.4f} seconds\n"
                f"Total Processing Time: {total_processing_time:.4f} seconds\n"
            )

            if os.path.isdir(output_path):
                # Output is a directory, write to default output.txt
                output_file_name = os.path.basename(input_path).replace(".txt", "_output.txt") if os.path.basename(input_path).endswith(".txt") else os.path.basename(input_path) + "_output.txt"
                output_file_path = os.path.join(output_path, output_file_name)
                try:
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        f.write(output_text)
                    print(f"Results for {os.path.basename(input_path)} have been written to {output_file_path}")
                except Exception as e:
                    print(f"Error occurred while writing to the output file: {e}")
            elif os.path.isfile(output_path) or not os.path.exists(output_path):
                # Output is a file or doesn't exist, write to it
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(output_text)
                    print(f"Results for {os.path.basename(input_path)} have been written to {output_path}")
                except Exception as e:
                    print(f"Error occurred while writing to the output file: {e}")
            else:
                print(f"Error: Invalid output path '{output_path}'. It should be an existing file or directory.")

        except FileNotFoundError:
            print(f"Error: Input file not found at {input_path}")
        except ValueError as ve:
            print(f"Error reading input file: {ve}")
        except Exception as e:
            print(f"An error occurred: {e}")

    elif os.path.isdir(input_path):
        # Input is a directory
        if os.path.isfile(output_path) or not os.path.exists(output_path):
            # Output is a single file, append all results
            try:
                with open(output_path, "w", encoding="utf-8") as outfile:
                    for filename in os.listdir(input_path):
                        if filename.endswith(".txt"):
                            input_file_path = os.path.join(input_path, filename)
                            print(f"\nProcessing file: {input_file_path}")
                            try:
                                with open(input_file_path, "r", encoding="utf-8") as infile:
                                    chinese_text = infile.read()
                                if not chinese_text.strip():
                                    print(f"Warning: Input file '{input_file_path}' is empty or contains only whitespace, skipping.")
                                    continue

                                total_start_time = time.time()
                                translate_start_time = time.time()
                                english_translation, translation_time = translate_zh_to_en_with_segmentation(chinese_text)
                                translate_end_time = time.time()
                                overall_translation_time = translate_end_time - translate_start_time

                                summarize_start_time = time.time()
                                english_summary = summarize_english_text(english_translation)
                                summarize_end_time = time.time()
                                summarization_time = summarize_end_time - summarize_start_time

                                total_end_time = time.time()
                                total_processing_time = total_end_time - total_start_time

                                chinese_char_count = len(chinese_text)
                                english_word_count = len(english_summary.split())

                                output_text = (
                                    f"--- Processing file: {filename} ---\n"
                                    f"English Summary:\n{english_summary}\n\n"
                                    f"Original Chinese Character Count: {chinese_char_count}\n"
                                    f"Summary English Word Count: {english_word_count}\n"
                                    f"Translation Time: {translation_time:.4f} seconds\n"
                                    f"Summarization Time: {summarization_time:.4f} seconds\n"
                                    f"Total Processing Time: {total_processing_time:.4f} seconds\n\n"
                                )
                                outfile.write(output_text)
                                print(f"Results for {filename} appended to {output_path}")

                            except Exception as e:
                                print(f"Error processing {filename}: {e}")
                print(f"\nAll results have been written to {output_path}")
            except Exception as e:
                print(f"Error writing to output file: {e}")

        elif os.path.isdir(output_path):
            # Output is a directory, write each file's result to a separate file
            for filename in os.listdir(input_path):
                if filename.endswith(".txt"):
                    input_file_path = os.path.join(input_path, filename)
                    output_filename = os.path.splitext(filename)[0] + "_output.txt"
                    output_file_path = os.path.join(output_path, output_filename)
                    print(f"\nProcessing file: {input_file_path}")
                    try:
                        with open(input_file_path, "r", encoding="utf-8") as infile:
                            chinese_text = infile.read()
                        if not chinese_text.strip():
                            print(f"Warning: Input file '{input_file_path}' is empty or contains only whitespace, skipping.")
                            continue

                        total_start_time = time.time()
                        translate_start_time = time.time()
                        english_translation, translation_time = translate_zh_to_en_with_segmentation(chinese_text)
                        translate_end_time = time.time()
                        overall_translation_time = translate_end_time - translate_start_time

                        summarize_start_time = time.time()
                        english_summary = summarize_english_text(english_translation)
                        summarize_end_time = time.time()
                        summarization_time = summarize_end_time - summarize_start_time

                        total_end_time = time.time()
                        total_processing_time = total_end_time - total_start_time

                        output_text = (
                            f"English Summary:\n{english_summary}\n\n"
                            f"Original Chinese Character Count: {len(chinese_text)}\n"
                            f"Summary English Word Count: {len(english_summary.split())}\n"
                            f"Translation Time: {translation_time:.4f} seconds\n"
                            f"Summarization Time: {summarization_time:.4f} seconds\n"
                            f"Total Processing Time: {total_processing_time:.4f} seconds\n"
                        )
                        with open(output_file_path, "w", encoding="utf-8") as outfile:
                            outfile.write(output_text)
                        print(f"Results for {filename} written to {output_file_path}")

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        else:
            print(f"Error: Invalid output path '{output_path}'. It should be an existing file or directory.")
    else:
        print(f"Error: Invalid input path '{input_path}'. It should be an existing file or directory.")