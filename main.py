import os
import argparse
import pandas as pd

from evaluate import evaluate
from utils import import_data_from_json
from text_summarizer_RAG import TextSummarizerRAG

def main(args):
    if args.config_file_path is None: 
        raise ValueError("Config file for model is missing. Please provide a valid configuration.")
    
    text_summarizer = TextSummarizerRAG(config_file_path=args.config_file_path)

    if args.evaluation_mode: 
        if args.data_test_path is None: 
            raise ValueError("Evaluation data is missing. Please provide a valid dataset.")
        
        test_dataset = import_data_from_json(args.data_test_path)
        score_rouge, score_sim, perf_dict = evaluate(test_dataset, text_summarizer)

        if args.output_folder is not None: 
            pd.DataFrame(perf_dict['Rouge']).to_csv(os.path.join(args.output_folder, "rouge_score.csv"))
            pd.DataFrame(perf_dict['Similarity']).to_csv(os.path.join(args.output_folder, "sim_score.csv"))

            score = f"Score rouge is {score_rouge}. \n Score sim is {score_sim}."
            text_file = open(os.path.join(args.output_folder, "score.txt"))
            text_file.write(score)
            text_file.close()

        else: 
            print(pd.DataFrame(perf_dict['Rouge']))
            print("\n")

            print(pd.DataFrame(perf_dict['Similarity']))
            print("\n")

            print(f"Score rouge is {score_rouge}")
            print("\n")

            print(f"Score sim is {score_sim}")
            print("\n")

    else:
        input_text = input("What do you want to summarize ? \n")

        summary = text_summarizer(input_text)

        dash_line = '-'.join('' for _ in range(100))
        print(dash_line)
        print(f'INPUT PROMPT:\n{input_text}')
        print(dash_line)
        print(f'MODEL SUMMARY:\n{summary}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TextSummarization", 
        description="This program load a text summarizer and test it on the data provided",
    )
    parser.add_argument(
        "--config_file_path", 
        type=str, 
        help="The path of the config file"
    )
    parser.add_argument(
        "--data_test_path", 
        type=str, 
        help="The path to the data that need to be tested"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        help="The folder in which to put the result"
    )
    parser.add_argument(
        "--evaluation_mode", 
        action="store_true",
        type=bool, 
        help="Whether or not to use the model for evaluation according to the hackathon template. "
    )

    main(parser.parse_args())