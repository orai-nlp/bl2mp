import os
import json
from argparse import ArgumentParser

from transformers import AutoTokenizer #, AutoModelForMaskedLM
from minicons import scorer

'''
#https://github.com/kanishkamisra/minicons
from minicons import scorer
mlm_model = scorer.MaskedLMScorer('orai-nlp/ElhBERTeu', 'cpu')
print(mlm_model.sequence_score([Kaixo Mundua!], reduction = lambda x: -x.sum(0).item()))
'''

def mlm_score(input_path, out_dir, lm, device, verbose):

	file_name = input_path.split("/")[-1]
	dataset_name = input_path.split("/")[-2]

	if lm[-1] == "/":
		lm_name = lm.split("/")[-2]
	else:
		lm_name = lm.split("/")[-1]

	out_path = out_dir+"/"+lm_name+"/"+dataset_name

	os.makedirs(out_path, exist_ok=True)

	#Init MODEL

	tokenizer = AutoTokenizer.from_pretrained(lm, do_lower_case=False) #, is_pretokenized=True)
	#model = AutoModelForMaskedLM.from_pretrained(lm)

	if device:
		mlm_model = scorer.MaskedLMScorer(lm, device)
	else:
		mlm_model = scorer.MaskedLMScorer(lm, device)

	t = 0
	a = 0

	with open(input_path, 'r') as json_file:
		json_list = list(json_file)
		with open(out_path+"/"+file_name, 'w') as out_file:

			for json_str in json_list:
				try:
					bl2mp = json.loads(json_str)
					sentence_good = bl2mp["sentence_good"]
					sentence_bad = bl2mp["sentence_bad"]
					bl2mp_type = bl2mp["type"]

					score_good, score_bad = mlm_model.sequence_score([sentence_good, sentence_bad], reduction = lambda x: -x.sum(0).item())

					token_good = tokenizer.tokenize(sentence_good)
					token_bad = tokenizer.tokenize(sentence_bad)

					if verbose:
						print("SENTENCE GOOD")
						print(sentence_good)
						print(score_good)
						print(token_good)
						print("SENTENCE BAD")
						print(sentence_bad)
						print(score_bad)
						print(token_bad)

					if len(token_good) == len(token_bad) and sentence_good != sentence_bad and sentence_good != "" and sentence_bad != "":
						if(score_good < score_bad):
							a+= 1
						t+=1

					probs = {"good": score_good, "bad": score_bad}
					tokenized = [[token_good], [token_bad]]

					bl2mp['probs'] = probs
					bl2mp['tokenized'] = tokenized
					#print((json.dumps(bl2mp) + os.linesep))
					out_file.write(json.dumps(bl2mp) + os.linesep)
				except:
					pass #print("ERROR")
					#print(sentence_good)
					#print(sentence_bad)


	print("ACC:")
	print(f"{a}/{t}")
	print(a/t*100)

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--input', type=str, required=True,
						help='Path to the input jsonl (directory or file).')
	parser.add_argument('--output_dir', type=str, required=True,
						help='Path to the output jsonl (directory or file).')
	parser.add_argument('--lm', type=str, required=True,
						help='Type of language model (identifier)')
	parser.add_argument('--device', type=str,
						help='choose cpu or cuda:{0, 1, ...}')
	parser.add_argument('--verbose', action='store_true',
						help='Whether or not to log verbosely.')
	args = parser.parse_args()

	mlm_score(input_path=args.input, out_dir=args.output_dir, lm=args.lm, device=args.device, verbose=args.verbose)
