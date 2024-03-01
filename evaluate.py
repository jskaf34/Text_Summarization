import numpy as np

from datasets import Dataset
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity_scores_text(text_ref : str, text_2 : str) -> float:
    reference_embedding = semantic_model.encode(text_ref, convert_to_tensor=True)
    sentence_embedding = semantic_model.encode(text_2, convert_to_tensor=True)
    return util.cos_sim(reference_embedding, sentence_embedding).cpu()

def compute_rouge(test_dataset : Dataset, model) -> dict[str, dict[str, str]]:
    scores = []
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(
        metrics,
        use_stemmer=True
    )
        
    for i in range(len(test_dataset)):
        example = test_dataset[i]  

        original_text = example['original_text']
        reference_summary = example['reference_summary']
        _, generated_summary = model(original_text)

        scores.append(
              scorer.score(
                    generated_summary,
                    reference_summary
              )
        )

    final_scores = {
        metric: {
            "precision": str(np.mean(
                [score.get(metric).precision for score in scores]
            )),
            "recall": str(np.mean(
                [score.get(metric).recall for score in scores]
            )),
            "fmeasure": str(np.mean(
                [score.get(metric).fmeasure for score in scores]
            )),
            }
        for metric in metrics
    }
    return final_scores

def compute_similarity_scores(test_dataset : Dataset, model) -> dict[str, dict[str, str]]:
    scores_with_reference = []
    scores_with_original_text = []

    for i in range(len(test_dataset)):
        example = test_dataset[i]  

        original_text = example['original_text']
        reference_summary = example['reference_summary']
        _, generated_summary = model(original_text)

        reference_embeddings = semantic_model.encode(
            reference_summary,
            convert_to_tensor=True
        )
        original_text_embeddings = semantic_model.encode(
            original_text,
            convert_to_tensor=True
        )
        generated_embeddings = semantic_model.encode(
            generated_summary,
            convert_to_tensor=True
        )

        scores_with_reference.append(
            util.cos_sim(
                reference_embeddings,
                generated_embeddings).cpu()
        )
        scores_with_original_text.append(
            util.cos_sim(
                original_text_embeddings,
                generated_embeddings).cpu()
        )
            
    final_scores = {
        "similarity_with_reference_summary": {
            "mean":   str(np.mean(scores_with_reference)),
            "median": str(np.median(scores_with_reference)),
            "std":    str(np.std(scores_with_reference))
        },
        "similarity_with_original_text": {
            "mean":   str(np.mean(scores_with_original_text)),
            "median": str(np.median(scores_with_original_text)),
            "std":    str(np.std(scores_with_original_text))
        }
    }
    return final_scores

def evaluate(test_dataset : Dataset, model):
    perf_dict = {
        "Rouge": compute_rouge(test_dataset, model),
        "Similarity": compute_similarity_scores(test_dataset, model)
    }
    score_rouge, score_sim = 0, 0

    performance_rouge = perf_dict['Rouge']
    performance_sim = perf_dict['Similarity']

    for metric in performance_rouge.values():
        score_rouge += float(metric['fmeasure'])

    score_sim = (2*float(performance_sim['similarity_with_reference_summary']['mean'])+float(performance_sim['similarity_with_original_text']['mean'])) / 3

    return score_rouge / 3, score_sim, perf_dict