"""
HEAD-QA v2 environment 

This script defines an evaluation environment for HEAD-QA v2 compatible with the Verifiers framework.

The `head_qa_v2_prompt` function is adapted from the HEAD-QA v2 paper (zero-shot prompting, Figure 6).

License:
  MIT License, Copyright (c) 2019 DVC

Citation:

@inproceedings{vilares-gomez-rodriguez-2019-head,
    title = "{HEAD}-{QA}: A Healthcare Dataset for Complex Reasoning",
    author = "Vilares, David  and
      G{\'o}mez-Rodr{\'i}guez, Carlos",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1092",
    doi = "10.18653/v1/P19-1092",
    pages = "960--966",
    abstract = "We present HEAD-QA, a multi-choice question answering testbed to encourage research on complex reasoning. The questions come from exams to access a specialized position in the Spanish healthcare system, and are challenging even for highly specialized humans. We then consider monolingual (Spanish) and cross-lingual (to English) experiments with information retrieval and neural techniques. We show that: (i) HEAD-QA challenges current methods, and (ii) the results lag well behind human performance, demonstrating its usefulness as a benchmark for future work.",
}
"""

from typing import Any

import verifiers as vf
from datasets import load_dataset
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer


def head_qa_v2_prompt(example: dict[str, Any]) -> dict[str, Any]:
    """Build the standard HEAD-QA v2 multiple-choice question prompt."""
    question_text = (example.get("qtext") or "").strip()
    answers = example.get("answers", [])
    options_text = "\n".join([f"{a['aid']}. {a['atext'].strip()}" for a in answers])
    
    prompt = (
        "You are an expert in specialized scientific and health disciplines. "
        "Respond to the following multiple-choice question:\n"
        "Provide the answer in the following JSON format: {Answer: [number]}.\n"
        "For example, if the answer is 1, write: {Answer: 1}\n\n"
        f"{question_text}\n{options_text}\n"
    )
    
    correct_answer = example.get("ra", -1)
    
    result = {
        "question": prompt,
        "answer": str(correct_answer),
        "choices": [str(a["aid"]) for a in answers],
        "gold_index": correct_answer - 1,
        "info": {"answer_text": answers[correct_answer - 1]["atext"].strip()},
    }
    return result 


def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict[str, Any] | None = None) -> float:
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text") if info else None
    is_correct = multiple_choice_accuracy(
        llm_answer=parsed,
        answer_letter=answer,
        answer_text=answer_text
    )
    return 1.0 if is_correct else 0.0


def load_environment(
    use_think: bool = False,
    system_prompt: str | None = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
) -> vf.Environment:
    '''
    Load the HEAD-QA v2 environment.
    Supports reasoning (use_think=True) or standard evaluation.
    Returns a SingleTurnEnv ready for model evaluation.
    '''
    val_ds = load_dataset("alesi12/head_qa_v2", 'en', split="train")
    
    def _map_example(example: dict[str, Any]) -> dict[str, Any] | None:
        correct_answer = example.get("ra", -1)
        question_text = (example.get("qtext") or "").strip()
        answers = example.get("answers", [])
        
        if not question_text or not answers or not (1 <= correct_answer <= len(answers)):
            return None
        
        options = [a["atext"].strip() for a in answers]
        answer_idx = correct_answer - 1
        
        if shuffle_answers:
            indices = [str(i+1) for i in range(len(options))]
            shuffled_options, _, answer_idx = randomize_multiple_choice(
                options=options,
                answer_choice=answer_idx,
                labels=indices,
                seed=shuffle_seed,
                row_id=question_text,
            )
            options = shuffled_options
    
        temp_example = {
            "qtext": question_text,
            "answers": [{"aid": i + 1, "atext": opt} for i, opt in enumerate(options)],
            "ra": answer_idx + 1
        }
        
        mapped = head_qa_v2_prompt(temp_example)
        return mapped
    
    
    columns_to_remove = ["qtext", "answers", "ra"]
    # Disable the Datasets cache when shuffling answers
    load_from_cache_file = False if shuffle_answers else True
    val_mapped = val_ds.map(_map_example,
                            remove_columns=columns_to_remove,
                            load_from_cache_file=load_from_cache_file,
    ).filter(lambda x: x is not None, load_from_cache_file=load_from_cache_file)
    
    # normalize answer_format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format

    if answer_format == AnswerFormat.XML:
        system_prompt = system_prompt or (THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT)
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        system_prompt = system_prompt or (THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT)
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")
    
    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=None,
        eval_dataset=val_mapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env