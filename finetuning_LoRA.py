import os
import yaml
import torch
import numpy as np

from trl import SFTTrainer
from peft import LoraConfig
from datasets import Dataset
from utils import import_data_from_json
from utils import prompt_instruction_format, prompt_instruction_format_t5
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TrainingArguments



def train_models(model_name, model, data, config_dict, formatting_fn):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #setting padding instructions for tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config_dict['parameters_ft']['output_dir'] = os.path.join(config_dict['parameters_ft']['output_dir'], model_name)
    
    trainingArgs = TrainingArguments(**config_dict['parameters_ft'])
    peft_config = LoraConfig(**config_dict['parameters_LoRA'])

    trainer = SFTTrainer(
        model=model,
        train_dataset=data['train'],
        eval_dataset = data['val'],
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=formatting_fn,
        args=trainingArgs,
        max_seq_length=512
    )

    trainer.train()

    return model

if __name__ == "__main__": 
    device = torch.device('cuda')

    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    airbus_datapath = os.path.join("./data/", "airbus_helicopters_train_set.json")
    train_dataset, val_dataset, test_dataset = import_data_from_json(airbus_datapath)

    for model_name in config['models']['models_names']: 
        print(f'Fine tuning {model_name} : \n')
        if "bart" in model_name: 
            formatting_fn = prompt_instruction_format_t5
        else: 
            formatting_fn = prompt_instruction_format
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        train_models(model_name, model, {"train": train_dataset, "val":val_dataset}, device, config)