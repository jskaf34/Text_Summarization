import yaml
import torch

from loguru import logger
from utils import compute_similarity_scores_text
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextSummarizer(): 
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as file:
            try:    
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        logger.info("Config info loaded")

        # With cuda
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # With MPS (Mac Silicon)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        ### Partie LLM Summarization
        self.n_brut_force = self.config['parameters_textsum_archi']['n_brut_force']
        models_params = self.config['models']['models_params']
        self.models_no_RAG = self.create_models_out_of_RAG(models_params, self.device)
        logger.info("Models loaded")

        ### Partie LLM + RAG
        datapath = self.config['dbRAG']['datapath']
        # Load RAG with LangChain

    @staticmethod
    def create_models_out_of_RAG(models_params : list[tuple], device): 
        list_models = []

        for tokenizer_name, model_path in models_params:

            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            list_models.append((tokenizer, model))
        
        return list_models
    
    def __call__(self, text_to_summarize):
        return self.predict_no_RAG(text_to_summarize)[1]

    def predict_no_RAG(self, text_to_summarize):
        list_output = []

        for tokenizer, model in self.models_no_RAG:
            model.eval()
            model.to(self.device)

            prompt = f"""
            {text_to_summarize}
            """
                
            inputs = tokenizer(prompt, return_tensors='pt')
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            list_output_model = [] 
            for _ in range(self.n_brut_force): 
                with torch.no_grad(): 
                    output_ids = model.generate(**inputs)
                    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                sim_score_output = compute_similarity_scores_text(text_to_summarize, output)

                if sim_score_output>0.85: 
                    return (sim_score_output, output)
                
                list_output_model.append((sim_score_output, output))

            list_output.append(max(list_output_model, key=lambda x: x[0]))

        return prompt, max(list_output, key=lambda x: x[0])
    
    def predict_with_RAG(self, text_to_summarize): 
        sim_score_output, output = self.predict_no_RAG(text_to_summarize)

        if sim_score_output > 0.85: 
            return output
        
        else: 
            ### Partie LLM + RAG : Enrichissement du résumé
            pass
