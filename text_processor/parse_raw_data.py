"""
Parse raw data from:
https://huggingface.co/datasets/andreuka18/pair_preference_eedi_llama
https://huggingface.co/datasets/andreuka18/pair_preference_eedi_mistral
"""
import re
import json
import pandas as pd


TEMPLATE = (
    "Question: {question}\n\nConstruct Name: {brief_description}\n\n"
    "Correct Answer: {correct_answer}\n\nIncorrect Answer: {incorrect_answer}"
)


def parse_data(raw_data, mode="llama"):
    assert mode in ["llama", "mistral"], "Invalid mode. Choose between 'llama' and 'mistral'."

    brief_description_pattern = re.compile(r"Brief description of the concept or skill being tested: (.*?)Question:", re.DOTALL)
    question_pattern = re.compile(r"Question: (.*?)Correct answer:", re.DOTALL)
    correct_answer_pattern = re.compile(r"Correct answer: (.*?)Incorrect answer:", re.DOTALL)

    if mode == "llama":
        incorrect_answer_pattern = re.compile(r"Incorrect answer: (.*?)What is the misconception", re.DOTALL)
        response_a_pattern = re.compile(r"\[RESPONSE A\] (.*?)\[RESPONSE B\]", re.DOTALL)
        response_b_pattern = re.compile(r"\[RESPONSE B\] (.*)", re.DOTALL)
    elif mode == "mistral":
        incorrect_answer_pattern = re.compile(r"Incorrect answer: (.*?)\[MISCONCEPTION A\]", re.DOTALL)
        response_a_pattern = re.compile(r"\[MISCONCEPTION A\] (.*?)\[MISCONCEPTION B\]", re.DOTALL)
        response_b_pattern = re.compile(r"\[MISCONCEPTION B\] (.*)", re.DOTALL)       
    
    parse_data = []

    for item in raw_data:
        text = item[0]["content"]
        preferences = item[1]["content"]

        brief_description = brief_description_pattern.search(text).group(1).strip()
        question = question_pattern.search(text).group(1).strip()
        correct_answer = correct_answer_pattern.search(text).group(1).strip()
        incorrect_answer = incorrect_answer_pattern.search(text).group(1).strip()
        winner = response_a_pattern.search(text).group(1).strip()
        loser = response_b_pattern.search(text).group(1).strip()

        if preferences == "B": winner, loser = loser, winner
        meta_data = {
            "brief_description": brief_description,
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }

        parse_data.append({
            "query": TEMPLATE.format_map(meta_data),
            "pos": [winner],
            "neg": [loser],
        })

    return parse_data


if __name__ == "__main__":
    dest_path = "data/pair_preference_eedi_llama.json"
    raw_data = pd.read_parquet("data/pair_preference_eedi_llama.parquet")
    mode = "llama"

    raw_data = raw_data["messages"].to_list()
    parse_data = parse_data(raw_data, mode)
    with open(dest_path, "w") as file:
        json.dump(parse_data, file, indent=4)