# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/eval/eval_for_gsm8k/eval.py
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

N_SHOT = 6
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "The answer is"

import time
from utils import get_final_norm, set_stop_words
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    set_seed
)
from llm import LLM, CustomRegressionLogitsProcessor
torch.set_grad_enabled(False)

def load_jsonl(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               is_gzip=False,
               num_samples=-1,
               random_seed=42):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None)
            item = new_item
            list_data_dict.append(item)
    if num_samples > 0:
        random.seed(random_seed)
        list_data_dict = random.sample(list_data_dict, num_samples)
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
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

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
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:" #[{"role": "user", "content": input_text}]
    return input_text_prompt

def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B", required=True)
    parser.add_argument("--mode", type=str, default="base", required=True)
    parser.add_argument("--dola-layer", type=str, default=None, choices=["high", "low"])
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["truthfulqa", "triviaqa", "natural_questions", "gsm8k"])
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--L", type=float, default=32)
    parser.add_argument("--step", type=float, default=1)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--n-shot", type=int, default=6)
    parser.add_argument("--do-sample", type=str, default=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1)

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
        file_name = "train.jsonl"
    else:
        assert args.num_samples == -1
        print("test mode")
        file_name = "test.jsonl"

    fp = os.path.join(dataset, file_name)
    if not os.path.exists(fp):
        download_url(
            'https://raw.githubusercontent.com/openai/'
            'grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/'
            f'grade_school_math/data/{file_name}', dataset)
        os.rename(os.path.join(dataset, file_name), fp)

    list_data_dict = load_jsonl(fp, instruction='question', output='answer', num_samples=args.num_samples, random_seed=args.random_seed)
    
    if args.debug:
        print("Debug mode")
        list_data_dict = list_data_dict[:10]

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
    results_dict = {'is_correct': [], 'model_answer': [], "full_text": [], 'model_completion': [], "time": [], "num_gen_token": []}
    with tqdm(total=len(list_data_dict)) as pbar:
        for sample in list_data_dict:
            set_seed(42)
            input_text = build_prompt(sample['instruction'], n_shot=args.n_shot, cot_flag=COT_FLAG)
            start = time.time()
            model_completion, gen_token = llm.gen(input_text, max_new_tokens=256, generate_kwargs=generate_kwargs)
            end = time.time()

            for stop_word in stop_word_list:
                length_to_remove = len(stop_word)
                if model_completion[-length_to_remove:] == stop_word:
                    model_completion = model_completion[:-length_to_remove]

            model_answer = clean_answer(model_completion)
            is_cor = is_correct(model_answer, sample['output'])
        
            answers.append(is_cor)
            results_dict["full_text"].append(input_text + model_completion)
            results_dict['is_correct'].append(is_cor)
            results_dict['model_answer'].append(model_answer)
            results_dict['model_completion'].append(model_completion)
            results_dict["time"].append(end - start)
            results_dict["num_gen_token"].append(gen_token)
            torch.cuda.empty_cache()

            pbar.update(1)
            pbar.set_description(f"Correct rate: {float(sum(answers))/len(answers) * 100:.2f}%,")

    results_dict['accuracy'] = np.mean(answers)
    output_dir = model_id.split("/")[1]
    output_file = mode if mode in ["base", "dola", "filter"] else f"M_{M}_L_{L}"
    output_file = mode + "_" + dola_layer if mode == "dola" else output_file
    
    if args.debug:
        output_file += "_debug"
        
    if args.valid and args.num_samples > 0:
        output_file += f"_samples_{args.num_samples}_seed_{args.random_seed}_valid"

    if not os.path.exists(os.path.join(dataset, output_dir)):
        os.makedirs(os.path.join(dataset, output_dir))

    with open(os.path.join(dataset, output_dir, output_file + f"_results_n{args.n_shot}_repe_{args.repetition_penalty}.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)