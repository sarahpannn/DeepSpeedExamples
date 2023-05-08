# copy from chatbot.py

import argparse
import re
import logging
import transformers  # noqa: F401
from transformers import pipeline, set_seed
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer
from datasets import load_dataset, load_metric


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
    set_seed(42)  # ???
    dataset = load_dataset("glue", "qqp")
    metric = load_metric("glue", "qqp")
    prompt = "Human: Does this question: "
    test_set = dataset['validation']
    label_table = {'yes': 1,
                   'no': 0, }

    predictions, labels = [], []

    for i in range(len(test_set)):
        response = get_model_response(generator, prompt,
                                      test_set['question1'][i] + " ask the same thing as this one: " +
                                      test_set['question2'][i] + "? Answer in a one-word, yes/no response. Assistant: ",
                                      max_new_tokens=args.max_new_tokens)

        prompt_len = len(prompt) + len(test_set['question1'][i] + " ask the same thing as this one: " +
                                       test_set['question2'][i] + "? Answer in a one-word, yes/no response. Assistant: ")
        
        print(response)

        response = response[0]['generated_text'][prompt_len:].split()[0].lower()
        try:
            label = label_table[response]
        except:
            label = -1

        predictions.append(label)
        labels.append(test_set[i]['label'])
        if i == 4:
            break

    correct = 0
    negatives = 0

    for i in range(len(predictions)):
        negatives += predictions[i] == -1
        correct += predictions[i] == labels[i]

    print(correct/len(predictions))
    print(negatives)


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    main(args)
