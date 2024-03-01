import os
import argparse
import pandas as pd

from evaluate import evaluate
from utils import import_data_from_json
from text_summarizer import TextSummarizer

def main(args):
    test_dataset = import_data_from_json(args.data_test_path)

    text_summarizer = TextSummarizer(config_file_path=args.config_file_path)

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
    main(parser.parse_args())