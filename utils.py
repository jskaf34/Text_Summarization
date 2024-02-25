import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split

def prompt_instruction_format(sample):
    return f"""### Instruction:
        Use the Task below and the Input given to write the Response:

        ### Task:
        Summarize the Input

        ### Input:
        {sample['original_text']}

        ### Response:
        {sample['reference_summary']}
        """ 

def import_data_from_json(datapath): 
    data_df = pd.read_json(datapath).T.reset_index()

    train_data, test_data = train_test_split(data_df, test_size=0.05)
    train_data, val_data = train_test_split(data_df, test_size=0.15)

    columns_to_remove = [key for key in train_data if key not in ("original_text", "reference_summary")]
    
    train_dataset = Dataset.from_pandas(train_data.drop(columns_to_remove, axis=1))
    val_dataset = Dataset.from_pandas(val_data.drop(columns_to_remove, axis=1))
    test_dataset = Dataset.from_pandas(test_data.drop(columns_to_remove, axis=1))

    return train_dataset.remove_columns('__index_level_0__'), val_dataset.remove_columns('__index_level_0__'), test_dataset.remove_columns('__index_level_0__')