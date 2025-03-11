from eval.squad_evaluate import *
from eval.utils import Question
from typing import Tuple
from utils.xformer import (get_response_from_HF_models, get_response_from_Meta_models,
						   create_input_messages_for_HF, create_input_messages_for_Meta)


def em(question: Question, candidate_answer: str, match: str = "string") -> int:
	return int(
		metric_max_over_ground_truths(
			regex_match if match == "regex" else exact_match_score,
			candidate_answer,
			question.answers,
		)
	)


def f1(question: Question, candidate_answer: str) -> float:
	return metric_max_over_ground_truths(
		f1_score,
		candidate_answer,
		question.answers,
	)


def open_llm(question: Question, candidate_answer: str, model_type: str, model, tokenizer) -> Tuple[int, str]:
	
	q_str = question.text
	answers = question.answers
	answers_str = "/".join(answers)
	# prompt = (f"Here is a question, a set of golden answers (split with /), an AI-generated answer. "
	# 		  f"Can you judge whether the AI-generated answer is correct according to the question and golden answers, "
	# 		  f"simply answer Yes or No."
	# 		  f"\n\nQuestion: {q_str}"
	# 		  f"\nGolden Answers: {answers_str}"
	# 		  f"\nAI-generated Answer: {candidate_answer}"
	# 		  f"\nA: ")
	system_prompt = "Please judge whether the AI-generated answer is correct according to the question and golden answers, simply answer Yes or No."
	prompt = f"Question: {q_str}\nGolden Answers: {answers_str}\nAI-generated Answer: {candidate_answer}\nA: "
	
	if model_type in [
		'Meta-Llama-3.1-8B-Instruct',
		'Meta-Llama-3.2-1B-Instruct',
		'Meta-Llama-3.2-3B-Instruct',
		'Meta-Llama-3.3-70B-Instruct',
	]:
		input_msgs = create_input_messages_for_Meta(system_prompt, prompt)
		response = get_response_from_Meta_models(input_msgs, model, tokenizer, max_new_tokens=10)
	else:
		input_msgs = create_input_messages_for_HF(system_prompt, prompt)
		response = get_response_from_HF_models(input_msgs, model, tokenizer, max_new_tokens=10)
	
	if 'yes' in response.lower() and 'no' not in response.lower():
		return 1, response
	elif 'no' in response.lower() and 'yes' not in response.lower():
		return 0, response
	else:
		return -1, response

		
