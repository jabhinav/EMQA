import os
import numpy as np
from transformers import GPT2Config, OpenAIGPTConfig, GPTNeoConfig, AutoConfig
from transformers import GPT2Tokenizer, OpenAIGPTTokenizer, AutoTokenizer
from transformers import OpenAIGPTLMHeadModel, GPTNeoForCausalLM, AutoModelForCausalLM, GPT2LMHeadModel
from typing import Optional

from utils.custom import is_rank_0


MODEL_CLASSES = {
	# ############################# OpenAI GPT Models ############################## #
	'gpt2': (GPT2Config, GPT2LMHeadModel),
	'gpt2-medium': (GPT2Config, GPT2LMHeadModel),
	'gpt2-large': (GPT2Config, GPT2LMHeadModel),
	'gpt2-xl': (GPT2Config, GPT2LMHeadModel),
	'gpt-neo-125M': (GPTNeoConfig, GPTNeoForCausalLM),
	'gpt-neo-1.3B': (GPTNeoConfig, GPTNeoForCausalLM),
	'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
}

TOKENIZER_CLASSES = {
	'gpt2': GPT2Tokenizer,
	'gpt2-medium': GPT2Tokenizer,
	'gpt2-large': GPT2Tokenizer,
	'gpt2-xl': GPT2Tokenizer,
	'gpt-neo-125M': GPT2Tokenizer,
	'gpt-neo-1.3B': GPT2Tokenizer,
	'openai-gpt': OpenAIGPTTokenizer,
}

LORA_IA3_TARGET_MODULES = {
	# ################################ OpenAI GPT Models ################################# #
	"gpt2": {
		"target_modules_lora": ["c_attn"],
	},
	"gpt2-medium": {
		"target_modules_lora": ["c_attn"],  # LoRA official targets only c_attn
	},
	"gpt2-large": {
		"target_modules_lora": ["c_attn"],
	},
	"gpt2-xl": {
		"target_modules_lora": ["c_attn"],
	},
	# ############################# Meta LLama Models ############################# #
	"Meta-Llama-3-8B": {
		"target_modules_lora": ["q_proj", "k_proj", "v_proj"],
	},
}

model_context = {
    "gpt-3.5-turbo-0125": {
        "context": 16385,
        "max_out": 4096
    },
    "gpt-4-turbo": {
        "context": 128000,
        "max_out": 4096
    },
    "HF-Llama-3-8B-Instruct": {
        "context": 4096,
        "max_out": 8192,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
    },
    "Meta-Llama-3.1-8B-Instruct": {
        "context": 4096,
        "max_out": 8192,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
    },
    "Meta-Llama-3.2-1B-Instruct": {
        "context": 4096,
        "max_out": 8192,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
    },
	"Meta-Llama-3.2-3B-Instruct": {
        "context": 4096,
        "max_out": 8192,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
    },
    "Meta-Llama-3.3-70B-Instruct": {
        "context": 4096,
        "max_out": 8192,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
    },
}


def get_model_size(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	model_size = sum([np.prod(p.size()) for p in model_parameters])
	return "{}M".format(round(model_size / 1e+6))


def load_tokenizer(model_type, model_path):
	if model_type in TOKENIZER_CLASSES:
		tokenizer_class = TOKENIZER_CLASSES[model_type]
	else:
		tokenizer_class = AutoTokenizer
		if is_rank_0():
			print("Using AutoTokenizer for model_type: ", model_type)
	
	tokenizer = tokenizer_class.from_pretrained(
		model_path,
		trust_remote_code=True,
		# token=HF_TOKEN,
	)
	
	if not tokenizer.eos_token:
		if tokenizer.bos_token:
			tokenizer.eos_token = tokenizer.bos_token
			tokenizer.eos_token_id = tokenizer.bos_token_id
			print("bos_token used as eos_token")
		else:
			raise ValueError("No eos_token or bos_token found")
	
	# Some Tokenizers do not have pad_token. We add it here. (It will only be used for ease of use in my pipeline.)
	if tokenizer.pad_token_id is None or tokenizer.pad_token is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id
		tokenizer.pad_token = tokenizer.eos_token
	
	if is_rank_0():
		print("Finish loading Tokenizer from %s", model_path)
	return tokenizer


def load_base_model(model_type, model_path, load_in_8bit: bool = False):
	if model_type in MODEL_CLASSES:
		config_class, model_class = MODEL_CLASSES[model_type]
	else:
		config_class, model_class = AutoConfig, AutoModelForCausalLM
	
	config = config_class.from_pretrained(
		model_path,
		trust_remote_code=True,
		revision="main",
		# token=HF_TOKEN,
	)
	model = model_class.from_pretrained(
		model_path,
		trust_remote_code=True,
		revision="main",
		# device_map="auto",
		# # For loading model in bfloat16, set. This is not quantization so it will not be as slow.
		# torch_dtype=torch.bfloat16,
		# # For loading model in 8bit, set. This is quantization so it will be slower.
		# load_in_8bit=True,
		# token=HF_TOKEN,
	)
	
	if is_rank_0():
		print("Finish loading Base model [%s] from %s", get_model_size(model), model_path)
	return config, model


def get_huggingface_path(model: str) -> str:
	# ############################# FacebookAI BERT Models ############################# #
	if model == 'roberta-large':
		path = 'roberta-large'  # roberta-large (355M)
	elif model == 'roberta-base':
		path = 'roberta-base'  # roberta-base (125M)
	# ############################# OpenAI GPT Models ############################# #
	elif model == 'gpt2':  # gpt2 (124M)
		path = 'gpt2'
	elif model == 'gpt2-medium':  # gpt2-medium(335M)
		path = 'gpt2-medium'
	elif model == 'gpt2-large':  # gpt2-large (774M)
		path = 'gpt2-large'
	elif model == 'gpt2-xl':
		path = 'gpt2-xl'  # gpt2-xl (1.5B)
	# ############################# EleutherAI GPT Models ############################# #
	elif model == 'gpt-neo-125M':
		path = 'EleutherAI/gpt-neo-125M'
	elif model == 'gpt-neo-1.3B':
		path = 'EleutherAI/gpt-neo-1.3B'  # 'EleutherAI/gpt-neo-1.3B' or 'EleutherAI/gpt-neo-2.7B'
	# ############################# Meta LLama Models ############################# #
	elif model == 'HF-Llama-3.2-1B-Instruct':
		path = 'meta-llama/Llama-3.2-1B-Instruct'
	# # [Specify the local path to model weights]
	elif model == 'Meta-Llama-3.1-8B-Instruct':
		path = '/my_code/ext_storage/llama_ckpts/Llama3.1-8B-Instruct'
	elif model == 'Meta-Llama-3.2-1B-Instruct':
		path = '/my_code/ext_storage/llama_ckpts/Llama3.2-1B-Instruct'
	elif model == 'Meta-Llama-3.2-3B-Instruct':
		path = '/my_code/ext_storage/llama_ckpts/Llama3.2-3B-Instruct'
	elif model == 'Meta-Llama-3.3-70B-Instruct':
		path = '/my_code/ext_storage/llama_ckpts/Llama3.3-70B-Instruct'
	# ############################# Alibaba Qwen Models ############################# #
	# Other Qwen models: 0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B
	elif model == 'Qwen2.5-0.5B-Instruct':
		path = 'Qwen/Qwen2.5-0.5B-Instruct'
	else:
		raise NotImplementedError()
	
	return path


def num_tokens_from_HF_models(text, tokenizer):
	num_tokens = 0
	for message in text:
		# Ensure the message content is a string
		content = str(message['content'])
		num_tokens += len(tokenizer.tokenize(content))
	return num_tokens


def load_HF_model(args):
	model_type: str = args.model_type
	model_path = get_huggingface_path(model_type)  # assuming get_huggingface_path is available in the current scope
	tokenizer = load_tokenizer(model_type, model_path)
	config, model = load_base_model(model_type, model_path)
	
	# # Specify to use the GPU
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#
	# # Move the model to the device
	# model = model.to(device)
	
	return tokenizer, config, model


def load_Meta_model(args):
	from llama_models.llama3.reference_impl.generation import Llama
	model_type: str = args.model_type
	model = Llama.build(
		ckpt_dir=get_huggingface_path(model_type),
		tokenizer_path=os.path.join(get_huggingface_path(model_type), "tokenizer.model"),
		max_seq_len=model_context[model_type]["context"],
		max_batch_size=args.max_batch_size,
		# model_parallel_size=args.nGPUs,  # We already specify this in torchrun cmd as --nproc_per_node
	)
	tokenizer = model.tokenizer
	config = model.args
	return tokenizer, config, model


def get_response_from_HF_models(
		dialog, gen_model, tokenizer, max_new_tokens=200, do_sample=False,
		top_k=50, top_p=0.95, num_return_sequences=1
):
	# Prepare the inputs and move them to the device
	inputs = tokenizer.apply_chat_template(dialog, add_generation_prompt=True, return_tensors="pt")
	inputs = inputs.to(gen_model.device)
	
	# tokenizer.eos_token_id is the id of <|EOT|> token
	outputs = gen_model.generate(
		inputs,
		max_new_tokens=max_new_tokens,
		do_sample=do_sample,
		top_k=top_k if do_sample else None,
		top_p=top_p if do_sample else None,
		num_return_sequences=num_return_sequences,
		eos_token_id=tokenizer.eos_token_id
	)
	return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


def get_response_from_Meta_models(
		dialog, gen_model, tokenizer=None, max_new_tokens=200, do_sample=False,
		top_k=50, top_p=0.95, num_return_sequences=1
):
	# # If using the generate method
	# # Prepare the inputs and move them to the device
	# prompt_tokens = [gen_model.formatter.encode_dialog_prompt(dialog)]
	# outputs, generation_logprobs = gen_model.generate(
	# 	prompt_tokens=prompt_tokens,
	# 	max_gen_len=max_new_tokens,
	# 	temperature=0.0,  # For deterministic generation
	# 	top_p=top_p,
	# 	logprobs=False,
	# 	echo=False,  # Do not echo the prompt in the response
	# )
	# response = tokenizer.decode(outputs[0])
	
	# # If using the chat_completion method
	response = gen_model.chat_completion(
		dialog,
		max_gen_len=max_new_tokens,
		temperature=0.0,
		top_p=None,
	)
	response = response.generation.content
	return response


def create_input_messages_for_HF(system_prompt: str, prompt: str):
	input_messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": prompt}
		]
	return input_messages


def create_input_messages_for_Meta(system_prompt: str, prompt: str):
	from llama_models.llama3.api.datatypes import RawMessage
	input_messages = [
		RawMessage(role="system", content=system_prompt),
		RawMessage(role="user", content=prompt),
	]
	return input_messages

