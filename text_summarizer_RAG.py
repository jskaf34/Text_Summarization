import re
import yaml
import torch


from loguru import logger
from utils import compute_similarity_scores_text
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader


class TextSummarizerRAG(): 
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
        self.api_key = self.config["api_key"]
        
        ### Partie LLM Summarization
        self.n_brut_force = self.config['parameters_textsum_archi']['n_brut_force']
        models_params = self.config['models']['models_params']
        self.models_no_RAG = self.create_models_out_of_RAG(models_params, self.device)
        logger.info("Models loaded")

        ### Partie RAG
        datapath = self.config['dbRAG']['datapath']
        if not datapath.endswith(".csv"):
            extension = datapath.split(".")[-1]
            raise TypeError(f"Data must be in csv format not {extension}")
        self.rag_params = self.config['modelsRAG']
        self.database = self.create_db_data(self.rag_params['chunk_size'], self.rag_params['chunk_overlap'])
        self.llm_RAG = HuggingFaceEndpoint(
            repo_id=self.config['repo_id'], max_length=128, temperature=0.3, token=self.rag_params['api_key']
        )

    @staticmethod
    def create_models(models_params : list[tuple], device): 
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
        return self.predict_with_RAG(text_to_summarize)

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
        output = self.predict_no_RAG(text_to_summarize)[1]

        if output[0] > 0.85: 
            return output[1]
        else: 
            ### Partie LLM + RAG : Enrichissement du résumé
            return self.RAG_execution(output[1], text_to_summarize)

    def RAG_execution(self, previous_summary, text_to_summarize):
        enhanced_summary, _ = self.get_summary_enhanced(previous_summary, text_to_summarize)
        return enhanced_summary
    
    def create_db_data(self, datapath, chunk_size, chunk_overlap) -> FAISS:
        embeddings = HuggingFaceEmbeddings()
        loader = CSVLoader(file_path=datapath)
        transcript = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(transcript)
        db = FAISS.from_documents(docs, embeddings)
        return db

    def get_summary_enhanced(self, summary, original_text,  k=3):
        docs = self.database.similarity_search(summary, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])

        # Use regular expression to remove ": {number}"
        cleaned_text = re.sub(r': \d{1,3}', '', docs_page_content)
        # Remove leading and trailing whitespaces
        cleaned_text = cleaned_text.strip()

        prompt = PromptTemplate(
            input_variables=["summary","original_text", "docs"],
            template="""
            You are a helpful legal assistant that that can summarize legal text.

            You have been asked to enhance the following summary: {summary}. The original text is: {original_text}. You can use the following documents to help you write in the same tone: {docs}. 
            Speak in a professional tone, as if you were writing a legal document.
            Do not mention the text. Be brief and shorter than the original text. 

            """
        )
        
        chain = LLMChain(llm=self.llm_RAG, prompt=prompt)
        response = chain.run(summary = summary, original_text = original_text, docs = cleaned_text)
        response = response.replace("\n", "")
        return response, docs