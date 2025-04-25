# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
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
import zipfile

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 6
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "So the answer is"
SHORT_ANSWER_TRIGGER = "answer is" # for long answer

import time
from utils import get_final_norm, set_stop_words
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    set_seed
)
from llm import LLM, CustomRegressionLogitsProcessor
torch.set_grad_enabled(False)

def load_jsonl(file_path, is_gzip=False):
    # Format of each line in StrategyQA:
    # {"qid": ..., "term": ..., "description": ..., "question": ..., "answer": ..., "facts": [...], "decomposition": [...]}
    
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        items = json.load(f)
        for item in items:
            new_item = dict(
                qid=item.get('qid', None),
                # term=item.get('term', None),
                # description=item.get('description', None),
                question=item.get('question', None),
                answer=item.get('answer', None),
                # facts=item.get('facts', []),
                # decomposition=item.get('decomposition', [])
            )
            list_data_dict.append(new_item)
    return list_data_dict

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text(n_shot=6, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("Do hamsters provide food for any animals?")
    chain.append("Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.")
    answer.append("yes")

    question.append("Could Brooke Shields succeed at University of Pennsylvania?")
    chain.append("Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.")
    answer.append("yes")

    question.append("Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?")
    chain.append("Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.")
    answer.append("no")

    question.append("Yes or no: Is it common to see frost during some college commencements?")
    chain.append("College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.")
    answer.append("yes")

    question.append("Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?")
    chain.append("The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.")
    answer.append("no")

    question.append("Yes or no: Would a pear sink in water?")
    chain.append("The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.")
    answer.append("no")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def build_prompt(input_text, n_shot, cot_flag, shuffle=False):
    demo = create_demo_text(n_shot, cot_flag, shuffle)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def clean_answer(model_pred, random_guess=True):
    model_pred = model_pred.lower()

    if "Thus, yes." in model_pred:
        preds = "yes"
    elif SHORT_ANSWER_TRIGGER.lower() in model_pred:
        preds = model_pred.split(SHORT_ANSWER_TRIGGER.lower())[1].split(".")[0].strip()
    else:
        #print("Warning: answer trigger not found in model prediction:", model_pred, "; returning yes/no based on exact match of `no`.", flush=True)
        if random_guess:
            preds = "no" if "no" in model_pred else "yes"
        else:
            return None
    if preds not in ["yes", "no"]:
        #print("Warning: model prediction is not yes/no:", preds, "; returning no", flush=True)
        if random_guess:
            preds = "no"
        else:
            return None

    return (preds == "yes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--mode", type=str, default="base", required=True)
    parser.add_argument("--dola-layer", type=str, default=None, choices=["high", "low"])
    parser.add_argument("--dataset", type=str, default="strqa", choices=["truthfulqa", "triviaqa", "natural_questions", "gsm8k", "strqa"])
    parser.add_argument("--data-path", type=str, default="strqa")
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--L", type=float, default=32)
    parser.add_argument("--step", type=float, default=1)
    parser.add_argument("--num-samples", type=int, default=229)
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
    data_path = args.data_path
    M = args.M
    L = args.L
    step = args.step

    assert mode in ["base", "extrapolation", "dola", "filter"], "Invalid method"

    print(f"model_id: {model_id}")
    print(f"dataset: {dataset}")
    print(f"method: {mode}")
    
    if args.valid:
        assert mode == "extrapolation", "Invalid mode for valid"
        assert args.num_samples > 0
        print("valid mode")
        print(f"num_samples: {args.num_samples}")
        print(f"random_seed: {args.random_seed}")
    else:
        print("test mode")

    

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to

    # Get test file
    '''
    The StrategyQA dataset includes the followings files:
        strategyqa_train.json: The training set of StrategyQA, which includes 2,290 examples.
        strategyqa_train_paragraphs.json: Paragraphs from our corpus that were matched as evidence for examples in the training set.
        strategyqa_train_filtered.json: 2,821 additional questions, excluded from the official training set, that were filtered by our solvers during data collection (see more details in the paper).
        strategyqa_test.json: The test set of StrategyQA, which includes 490 examples.
    Here we only need the test set.
    '''

    file_name = "strategyqa_train.json"
    fp = os.path.join(args.data_path, file_name)
    if not os.path.exists(fp):
        download_url(
            'https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip', args.data_path)

        # Once the file is downloaded, unzip it
        with zipfile.ZipFile(os.path.join(args.data_path, 'strategyqa_dataset.zip'), 'r') as zip_ref:
            zip_ref.extractall(args.data_path)

    list_data_dict = load_jsonl(fp)

    random.seed(args.random_seed)
    valid_idx = random.sample(range(len(list_data_dict)), args.num_samples)
    test_idx = [i for i in range(len(list_data_dict)) if i not in valid_idx]

    if args.valid:
        list_data_dict = [list_data_dict[i] for i in valid_idx]
    else:
        list_data_dict = [list_data_dict[i] for i in test_idx]
    
    if args.debug:
        print("Debug mode")
        list_data_dict = list_data_dict[:25]

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
            assert M, "M is required for extrapolation mode"
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

    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], "time": [], "num_gen_token": []}
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample['question'], N_SHOT, COT_FLAG)
        set_seed(42)
        start = time.time()
        model_completion, gen_token = llm.gen(input_text, max_new_tokens=256, generate_kwargs=generate_kwargs)
        end = time.time()
        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        model_completion = model_completion.strip()
        model_answer = clean_answer(model_completion, random_guess=True)
        is_cor = is_correct(model_answer, sample['answer'])

        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_answer'].append(model_answer)
        result_dict['model_completion'].append(model_completion)
        result_dict["time"].append(end - start)
        result_dict["num_gen_token"].append(gen_token)

    result_dict['accuracy'] = np.mean(answers)
    output_dir = model_id.split("/")[1]
    output_file = mode if mode in ["base", "dola", "filter"] else f"M_{M}_L_{L}"
    output_file = mode + "_" + dola_layer if mode == "dola" else output_file
    
    if args.debug:
        output_file += "_debug"
        
    if args.valid and args.num_samples > 0:
        output_file += f"_samples_{args.num_samples}_seed_{args.random_seed}_valid"

    if not os.path.exists(os.path.join(dataset, output_dir)):
        os.makedirs(os.path.join(dataset, output_dir))

    with open(os.path.join(dataset, output_dir, output_file + f"_results_repe_{args.repetition_penalty}.json"), 'w') as f:
        json.dump(result_dict, f, indent=4)