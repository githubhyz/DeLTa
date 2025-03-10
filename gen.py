# Ref: Dola
import re
import os
import json
import random
import torch
torch.set_grad_enabled(False)
import numpy as np
import transformers
from tqdm import tqdm
import argparse

import ssl
import urllib.request
import gzip

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "The answer is"

import time
from utils import get_final_norm, set_stop_words, build_prompt, load_csv
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    set_seed
)
from llm import LLM, CustomRegressionLogitsProcessor
torch.set_grad_enabled(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B", required=True)
    parser.add_argument("--mode", type=str, default="base", required=True)
    parser.add_argument("--dola-layer", type=str, default=None, choices=["high", "low"])
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=["truthfulqa", "triviaqa", "natural_questions", "gsm8k"])
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--L", type=float, default=32)
    parser.add_argument("--step", type=float, default=1)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--do-sample", type=str, default=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)

    args = parser.parse_args()

    model_id = args.model_id
    mode = args.mode
    dola_layer = args.dola_layer
    dataset = args.dataset
    M = args.M
    L = args.L
    step = args.step

    assert mode in ["base", "extrapolation", "dola", "filter"], "Invalid method"

    print(f"model_id: {model_id}")
    print(f"dataset: {dataset}")
    print(f"method: {mode}")
    
    if args.valid:
        assert mode == "extrapolation", "Invalid method for valid"
        assert args.num_samples > 0
        print("valid mode")
        print(f"num_samples: {args.num_samples}")
        print(f"random_seed: {args.random_seed}")
    else:
        assert args.num_samples == -1
        print("test mode")

    list_data_dict = load_csv('TruthfulQA.csv', dataset=dataset, num_samples=args.num_samples, random_seed=args.random_seed)

    if args.debug:
        print("Debug mode")
        list_data_dict = list_data_dict[:50]

    llm = LLM(model_id)
    model = llm.model
    device = llm.model.device
    N_layer = llm.model.config.num_hidden_layers
    stop_word_list = ["Q:"]
    stopping_criteria = set_stop_words(model, llm.tokenizer, stop_word_list)

    generate_kwargs = dict(
        num_return_sequences=1,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        stopping_criteria=stopping_criteria,
    )

    if mode == "dola":
        assert dola_layer in ["high", "low"], "Invalid dola layer"
        print(f"dola_layer: {dola_layer}")
        generate_kwargs['dola_layers'] = dola_layer

    elif mode == "extrapolation" or mode == "filter":
        if mode == "extrapolation":
            assert M >= 0 and isinstance(M, int), "M is required for extrapolation mode"
            assert N_layer <= L, "L should be greater than the number of layers in the model"
            print(f"M: {M}")
            print(f"L: {L}")
            print(f"step: {step}")

        X_reg = torch.arange(M, N_layer + 0.00001, step=step, device=device)  # (reg_range,)

        extrapolation_args = dict(
            mode=mode,
            model=model,
            device=model.device,
            lm_head=model.get_output_embeddings(),
            final_norm=get_final_norm(model),

            M=M,
            L=L,
            step=step,
            mean_X_reg=torch.mean(X_reg, dim=0),
            var_X_reg=torch.var(X_reg, unbiased=False, dim=0) + 1e-9,
            X_reg_=X_reg[:, None] - torch.mean(X_reg, dim=0),
        )
        
        generate_kwargs.pop("repetition_penalty")
        logits_processor = LogitsProcessorList([
            CustomRegressionLogitsProcessor(**extrapolation_args), 
            RepetitionPenaltyLogitsProcessor(args.repetition_penalty)
        ])
        generate_kwargs['logits_processor'] = logits_processor

    results_dict = []
    for sample in tqdm(list_data_dict):
        set_seed(42)
        if dataset == "truthfulqa":
            question = sample
        elif dataset == "triviaqa":
            question = sample["question"]
        elif dataset == "natural_questions":
            question = sample["question"]
            short_answers = sample["short_answers"]
            if not short_answers or short_answers == [""]:
                continue

        input_text = build_prompt(question, dataset=dataset)
        
        start = time.time()
        model_completion, gen_token = llm.gen(input_text, max_new_tokens=50, generate_kwargs=generate_kwargs)
        end = time.time()

        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        model_completion = model_completion.strip()

        if dataset == "truthfulqa":
            results_dict.append(
                {"index": len(results_dict), "question": input_text, "model_completion": model_completion, "time": end - start, "num_gen_token": gen_token}
            )
        elif dataset == "triviaqa":
            results_dict.append(
                {"index": len(results_dict), "question": input_text, "model_completion": model_completion, "answers": sample["answers"], "time": end - start, "num_gen_token": gen_token}
            )
        elif dataset == "natural_questions":
            results_dict.append(
                {"index": len(results_dict), "question": input_text, "model_completion": model_completion, "answers": ", ".join(sample["short_answers"]), "time": end - start, "num_gen_token": gen_token}
            )

    output_dir = model_id.split("/")[1]
    output_file = mode if mode in ["base", "dola", "filter"] else f"M_{M}_L_{L}"
    output_file = mode + "_" + dola_layer if mode == "dola" else output_file
    
    if args.debug:
        output_file += "_debug"
        
    if args.valid and args.num_samples > 0:
        output_file += f"_samples_{args.num_samples}_seed_{args.random_seed}_valid"

    if not os.path.exists(os.path.join(dataset, output_dir)):
        os.makedirs(os.path.join(dataset, output_dir))

    with open(os.path.join(dataset, output_dir, output_file + f"_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)