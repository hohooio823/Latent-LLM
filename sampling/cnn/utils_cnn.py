from rouge_score import rouge_scorer, scoring
import re

def split_sentences(sentences, keep=3):
    # Regular expression to split the sentence on punctuation marks followed by an optional quote and a space
    
    split_sentences = re.split(r'([.!?]["\']?)+', sentences)

    # Combine back the punctuation marks and split text
    final_sentences = [split_sentences[i] + split_sentences[i+1] for i in range(0, len(split_sentences)-1, 2)]
    summary = ''.join(final_sentences[:keep])

    if len(final_sentences) < keep:
        return sentences
    
    return summary


def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score)
    
    aggregator = scoring.BootstrapAggregator()
    for score in scores:
        aggregator.add_scores(score)
    
    result = aggregator.aggregate()
    return {
        'rouge1': result['rouge1'].mid.fmeasure * 100,
        'rouge2': result['rouge2'].mid.fmeasure * 100,
        'rougeL': result['rougeL'].mid.fmeasure * 100
    }
