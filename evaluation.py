from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score  # You might need to install bert-score: pip install bert-score


def evaluate_summary(reference, hypothesis):
    """
    Calculates ROUGE, BLEU, and BERT scores for a given summary. (Excluding BLEU to focus on summarization part)

    Args:
        reference (str): The reference (ground truth) English text.
        hypothesis (str): The generated English summary.

    Returns:
        dict: A dictionary containing ROUGE, BLEU, and BERT scores.
    """

    # ROUGE Scoring
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_obj.score(reference, hypothesis)

    # BERT Scoring
    P, R, F1 = score([hypothesis], [reference], lang='en')  # Returns tensors, convert to scalar

    return {  # THIS IS THE CRUCIAL LINE
        "rouge1_fmeasure": rouge_scores['rouge1'].fmeasure,
        "rouge1_precision": rouge_scores['rouge1'].precision,
        "rouge1_recall": rouge_scores['rouge1'].recall,
        "rougeL_fmeasure": rouge_scores['rougeL'].fmeasure,
        "rougeL_precision": rouge_scores['rougeL'].precision,
        "rougeL_recall": rouge_scores['rougeL'].recall,
        # "bleu": bleu,
        "bert_precision": P.item(),
        "bert_recall": R.item(),
        "bert_f1": F1.item()
    }