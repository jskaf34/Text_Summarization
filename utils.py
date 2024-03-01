import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

device = torch.device('mps')
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def prompt_instruction_format_t5(sample):
    return f"""### Instruction:
        Use the Task below and the Input given to write the Response:

        ### Task:
        Summarize the Input

        ### Input:
        {sample['original_text']}

        ### Response:
        {sample['reference_summary']}
        """ 

def prompt_instruction_format(sample): 
    return f""" 
    ### Input: 
    {sample['original_text']}

    ### Response:
    {sample['reference_summary']}
    """

def import_data_from_json(datapath): 
    data_df = pd.read_json(datapath).T.reset_index()

    data_df['text_length'] = data_df['original_text'].apply(len)

    max_length = data_df['text_length'].max()
    bins = [0, max_length*0.2, max_length*0.4, max_length*0.6, max_length*0.8, max_length]
    labels = ['1', '2', '3', '4', '5']
    data_df['length_category'] = pd.cut(data_df['text_length'], bins=bins, labels=labels)

    train_data, test_data = train_test_split(data_df, test_size=0.01, stratify=data_df['length_category'])
    train_data, val_data = train_test_split(data_df, test_size=0.18, stratify=data_df['length_category'])

    columns_to_remove = [key for key in train_data if key not in ("original_text", "reference_summary")]
    
    train_dataset = Dataset.from_pandas(train_data.drop(columns_to_remove, axis=1))
    val_dataset = Dataset.from_pandas(val_data.drop(columns_to_remove, axis=1))
    test_dataset = Dataset.from_pandas(test_data.drop(columns_to_remove, axis=1))

    return train_dataset.remove_columns('__index_level_0__'), val_dataset.remove_columns('__index_level_0__'), test_dataset.remove_columns('__index_level_0__')

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
