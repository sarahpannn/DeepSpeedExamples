# copy from chatbot.py

import argparse
import re
import logging
import transformers  # noqa: F401
from transformers import pipeline, set_seed
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer
from datasets import load_dataset, load_metric
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    return args


def get_generator(path):
    tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path)
    model = OPTForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config).half()

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         device="cuda:0")
    return generator


def get_model_response(generator, prompt, user_input, max_new_tokens):
    response = generator(prompt + user_input, max_new_tokens=max_new_tokens)
    return response


def process_response(response, num_rounds):
    output = str(response[0]["generated_text"])
    output = output.replace("<|endoftext|></s>", "")
    all_positions = [m.start() for m in re.finditer("Human: ", output)]
    place_of_second_q = -1
    if len(all_positions) > num_rounds:
        place_of_second_q = all_positions[num_rounds]
    if place_of_second_q != -1:
        output = output[0:place_of_second_q]
    return output


def main(args):
    generator = get_generator(args.path)
    # set_seed(42)  # ???
    dataset = load_dataset("glue", "qqp")
    dataset = dataset.shuffle()
    metric = load_metric("glue", "qqp")
    prefix = "Human: Do these two questions ask the same thing? "
    shot_one = "Why are computers so expensive? How expensive are computers? [y/n] \nAssistant: No. \n"
    shot_two = "What color is a banana? What shade of color does a banana have? [y/n] \nAssistant: Yes. \n"
    shot_three = "What is the capital of California? Where is the capital of Georgia? [y/n] \nAssistant: No. \n"
    shot_four = "How many markers do you have? What is the number of markers you have? [y/n] \nAssistant: Yes. \n"
    shot_five = "How should I write a book? Have you ever written a novel? [y/n] \nAssistant: No. \n"
    shot_six = "Can I get a decaf coffee? Can I get a coffee without caffeine? [y/n] \nAssistant: Yes. \n"
    prompt = prefix + shot_two + prefix + shot_one + prefix + shot_three + prefix + shot_four + prefix + shot_five + prefix + shot_six + prefix
    test_set = dataset['validation']

    predictions, labels = [], dataset['validation']['label']

    for i in range(len(tqdm(test_set))):
        response = get_model_response(generator, prompt,
                                      test_set['question1'][i] + " " +
                                      test_set['question2'][i] + " [y/n] \nAssistant: ",
                                      max_new_tokens=args.max_new_tokens)

        prompt_len = len(prompt) + len(test_set['question1'][i] + " " +
                                       test_set['question2'][i] + " [y/n] \nAssistant: ")
        
        if len(response[0]['generated_text'][prompt_len:]) > 0:
            try:
                # print(response[0]['generated_text']
                #       [prompt_len:].split()[0].lower())
                if "yes" in response[0]['generated_text'][prompt_len:].split()[0].lower():
                    predictions.append(1)
                elif "no" in response[0]['generated_text'][prompt_len:].split()[0].lower():
                    predictions.append(0)
                else:
                    predictions.append(-1)
            except:
                predictions.append(-1)
        if i == 1000:
            break
        
        # print(response[0]['generated_text'])
        # print("~~~~~~~~~~~~~~~~~~~~")
    
    correct = 0
    tot = 0
    for i in range(len(predictions)):
        correct += predictions[i] == labels[i]
        tot += predictions[i] != -1

    print(correct/tot)


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    main(args)
