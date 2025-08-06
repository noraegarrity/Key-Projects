from transformers import MarianTokenizer, MarianMTModel
import time

def translate_zh_to_en_with_segmentation(text, model_name='Helsinki-NLP/opus-mt-zh-en', max_tokens=512):
    """
    Translates Traditional Chinese text to English, handling segmentation for long texts.

    Args:
        text (str): The Traditional Chinese text to translate.
        model_name (str, optional): The name of the translation model to use.
        max_tokens (int, optional): The maximum number of tokens the model can handle at once.

    Returns:
        tuple: The translated English text and the total translation time.
    """

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