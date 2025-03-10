import os
import json
from typing import Dict, List, Optional, Tuple, Union
import re
import html
import random
import numpy as np
import torch
from datasets import load_dataset
import pandas as pd
import transformers
from transformers import models, AutoTokenizer
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    add_start_docstrings,
    STOPPING_CRITERIA_INPUTS_DOCSTRING
)
from demo_text import DEMO

Norm = Union[
    torch.nn.LayerNorm,
    models.llama.modeling_llama.LlamaRMSNorm,
    models.gemma2.modeling_gemma2.Gemma2RMSNorm,
    models.mistral.modeling_mistral.MistralRMSNorm,
    torch.nn.Module,
]

# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

# Ref: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/natural_qa_scenario.py
def _clean_text(raw_text: bytes):
    text = raw_text.replace(b"\xc2\xa0", b" ").decode("utf-8")
    return re.sub("<([^>]*)>", "", html.unescape(text))

def create_raw_instance(sample: Dict) -> Dict:
    html_bytes = sample["document"]["html"].encode("utf-8")
    question = sample["question"]["text"]
    ans_json = sample["annotations"]

    long_answers = set()
    for ans in ans_json["long_answer"]:
        start_byte = ans["start_byte"]
        end_byte = ans["end_byte"]
        if not start_byte and not end_byte:
            continue
        long_answers.add(_clean_text(html_bytes[ans["start_byte"] : ans["end_byte"]]))

    short_answers = set()
    for ans in ans_json["short_answers"]:
        start_byte = ans["start_byte"]
        end_byte = ans["end_byte"]
        if not start_byte and not end_byte:
            continue
        short_answers.add(_clean_text(html_bytes[ans["start_byte"][0] : ans["end_byte"][0]]))

    return {
        "question": question.capitalize(),
        "long_answers": list(long_answers),
        "short_answers": list(short_answers)
    }

# for datasets
def load_csv(file_path="", dataset='truthfulqa', num_samples=-1, random_seed=42):
    # input file is in csv format, can be loaded by pandas
    # required columns: [Question] only
    print("Loading dataset: ", dataset)
    #progress_bar = tqdm(total=0, dynamic_ncols=True)
    list_data = []
    if dataset == 'truthfulqa':
        assert file_path != "", "Please provide the file path for the dataset"
        with open(file_path, 'r') as f:
            df = pd.read_csv(f)
            list_data = list(df['Question'])
    elif dataset == "triviaqa":
        if num_samples > 0:
            split = "train"
        else:
            split = "validation"
        data = load_dataset("mandarjoshi/trivia_qa", 'unfiltered.nocontext', split=split, streaming=True)
        iterator = iter(data)
        while True:
            try:
                sample = next(iterator)
                answer = sample["answer"]
                answers = list(set(answer["aliases"] + answer["normalized_aliases"]))
                list_data.append({
                    "question": sample["question"],
                    "answers": answers
                })
                #progress_bar.update(1)
            except StopIteration:
                break
    elif dataset == "natural_questions":
        if num_samples > 0:
            split = "validation"
        else:
            split = "test"
        data = load_dataset("google-research-datasets/natural_questions", split="validation", streaming=True)
        iterator = iter(data)
        temp_data = []
        while True:
            try:
                sample = next(iterator)
                temp_data.append(sample)
            except StopIteration:
                break
        
        list_data  = create_raw_instance(temp_data)

    if num_samples > 0:
        random.seed(random_seed)
        list_data = random.sample(list_data, num_samples)

    print(f"Loaded {len(list_data)} samples from {dataset} dataset.")
    return list_data


def build_prompt(input_text, dataset='truthfulqa'):
    demo = DEMO[dataset]
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def build_prompt_triviaqa(input):
    return f"Answer the following question concisely.\nQ: {input}\nA:"

# for llm model
def get_final_norm(model: transformers.PreTrainedModel) -> Norm:
    base_model = model.base_model
    if isinstance(base_model, models.opt.modeling_opt.OPTModel):
        final_layer_norm = base_model.decoder.final_layer_norm
    elif isinstance(base_model, models.gpt_neox.modeling_gpt_neox.GPTNeoXModel):
        final_layer_norm = base_model.final_layer_norm
    elif isinstance(
        base_model,
        (
            models.bloom.modeling_bloom.BloomModel,
            models.gpt2.modeling_gpt2.GPT2Model,
            models.gpt_neo.modeling_gpt_neo.GPTNeoModel,
            models.gptj.modeling_gptj.GPTJModel,
        ),
    ):
        final_layer_norm = base_model.ln_f
    elif isinstance(
        base_model, 
        (
            models.llama.modeling_llama.LlamaModel,
            models.gemma2.modeling_gemma2.Gemma2Model,
            models.mistral.modeling_mistral.MistralModel,
            models.qwen2.modeling_qwen2.Qwen2Model,
        )
    ):
        final_layer_norm = base_model.norm
    else:
        raise NotImplementedError(f"Unknown model type {type(base_model)}")

    if final_layer_norm is None:
        raise ValueError("Model does not have a final layer norm.")

    assert isinstance(final_layer_norm, Norm.__args__)  # type: ignore
    return final_layer_norm
    

class CustomStoppingCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the model generates '\nQ:' tokens. It means that the model has finished generating the answer and start generating a new question.
    """
    def __init__(self, list_token_ids_sequence: list):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len(token_ids_sequence))
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # check the final {self.length} tokens
        stop = False
        for token_ids_sequence, length in zip(self.token_ids_sequences, self.lengths):
            if input_ids.shape[-1] < length:
                continue
            else:
                if bool(torch.all(input_ids[0, -length:] == token_ids_sequence.to(input_ids.device))):
                    stop = True
                    break
        return stop

    
def set_stop_words(model, tokenizer, stop_words: List[str]):
    base_model = model.base_model
    stopping_criteria = StoppingCriteriaList()
    list_stop_word_ids = []
    for stop_word in stop_words:
        if isinstance(
            base_model, 
            (
                models.llama.modeling_llama.LlamaModel,
                models.mistral.modeling_mistral.MistralModel,
            ),
        ):
            stop_word_ids = tokenizer.encode('\n' + stop_word)[3:]
        elif isinstance(
            base_model, models.gemma2.modeling_gemma2.Gemma2Model,
        ):
            stop_word_ids = tokenizer.encode('\n' + stop_word)[2:]
        elif isinstance(
            base_model, 
            (
                models.gptj.modeling_gptj.GPTJModel,
                models.gpt_neox.modeling_gpt_neox.GPTNeoXModel,
                models.qwen2.modeling_qwen2.Qwen2Model,
            ),
        ):
            stop_word_ids = tokenizer.encode('\n' + stop_word)[1:]
            
        list_stop_word_ids.append(stop_word_ids)
        print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)

    stopping_criteria.append(CustomStoppingCriteria(list_stop_word_ids))

    return stopping_criteria
