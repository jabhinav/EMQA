import argparse
import json
from eval.utils import Question
from eval.metrics import em, f1, open_llm
from utils.xformer import load_HF_model, load_Meta_model
from utils.custom import is_rank_0
from tqdm import tqdm

from huggingface_hub import login
# Read the hf_token from the file
path_to_file = './hf_key.txt'
with open(path_to_file, 'r') as file:
	hf_token = file.read().replace('\n', '')
login(token=hf_token)


def read_annotations_file(annotations_path):
	with open(annotations_path) as f:
		annotations = json.load(f)
	return annotations


def read_results_file(results_path):
	with open(results_path) as f:
		results = json.load(f)
	return results


def evaluate(args):
	annotations = read_annotations_file(args.annotations_path)
	results = read_results_file(args.results_path)

	eval_per_sample = {}
	em_scores, f1_scores = [], []
	open_llm_scores = []
	if args.do_open_llm_eval:
		if args.model_type in [
			'Meta-Llama-3.1-8B-Instruct',
			'Meta-Llama-3.2-1B-Instruct',
			'Meta-Llama-3.2-3B-Instruct',
			'Meta-Llama-3.3-70B-Instruct',
		]:
			tokenizer, config, model = load_Meta_model(args)
		else:
			tokenizer, config, model = load_HF_model(args)
		if is_rank_0():
			print(f"Loaded model: {args.model_type}")
	else:
		# To avoid errors while saving the results
		args.model_type = ''
	
	for video_id in tqdm(annotations, desc="Evaluating Videos", total=len(annotations), dynamic_ncols=True, position=0, unit="video", colour="GREEN", disable=not is_rank_0()):
		
		# Check if video_id is in results
		if video_id in results:
			
			samples = annotations[video_id]
			for sample in tqdm(samples, desc="Evaluating Samples", total=len(samples), dynamic_ncols=True, position=1, unit="sample", leave=False, colour="BLUE", disable=not is_rank_0()):
				
				sample_id = sample['sample_id']
				question = sample['question']
				golden_answers = [sample['answer'] if isinstance(sample['answer'], str) else sample['answer']]
				
				for result in results[video_id]:
					if result['sample_id'] == sample_id:
						
						candidate_answer = result['pred']
						
						# Evaluate
						question_obj = Question(question, golden_answers, sample_id)
						em_score = em(question_obj, candidate_answer)
						f1_score = f1(question_obj, candidate_answer)
						
						if args.do_open_llm_eval:
							open_llm_score, open_llm_response = open_llm(question_obj, candidate_answer, args.model_type, model, tokenizer)
						else:
							open_llm_score, open_llm_response = None, None
							
						# Store results
						em_scores.append(em_score)
						f1_scores.append(f1_score)
						# Don't consider samples where LLM could not reply a simple Yes/No
						if open_llm_score != -1:
							open_llm_scores.append(open_llm_score)
							
						eval_per_sample[sample_id] = {
							"video_id": video_id,
							"em": em_score,
							"f1": f1_score,
							"open_llm": open_llm_score,
							"open_llm_response": open_llm_response,
							"question": question,
							"golden_answers": golden_answers,
							"candidate_answer": candidate_answer
						}
						break
		else:
			if is_rank_0():
				print("Video not found in results!")
			
	# Calculate average scores
	average_em = sum(em_scores) / len(em_scores)
	average_f1 = sum(f1_scores) / len(f1_scores)
	average_open_llm = sum(open_llm_scores) / len(open_llm_scores) if args.do_open_llm_eval else None
	
	eval_results = {
		'scores': {
			'average/em': average_em,
			'average/f1': average_f1,
			f'average/open_llm/{args.model_type}': average_open_llm if args.do_open_llm_eval else None
		},
		'eval_per_sample': eval_per_sample
	}
	
	# Print results
	if is_rank_0():
		print(f"Average EM: {average_em}")
		print(f"Average F1: {average_f1}")
		print(f"Average Open-LLM({args.model_type}): {average_open_llm}")
		
		# Save results
		with open(f"eval_results_{args.model_type}.json", "w") as f:
			json.dump(eval_results, f, indent=4)
	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--annotations_path', default='../ext_storage/Ego4D/my_hierarchical-emv_qa.json')
	parser.add_argument('--results_path', default='../LongVU/results_cambrian_llama3.json')
	parser.add_argument('--do_open_llm_eval', default=True)
	parser.add_argument('--model_type', default="Meta-Llama-3.2-1B-Instruct")
	parser.add_argument('--max_batch_size', default=4, type=int)
	parser.add_argument('--nGPUs', default=8)
	args = parser.parse_args()

	evaluate(args)
	
	