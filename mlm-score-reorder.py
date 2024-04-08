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
	t2 = 0
	a2 = 0
	t3 = 0
	a3 = 0

	with open(input_path, 'r') as json_file:
		json_list = list(json_file)

		with open(out_path+"/"+file_name, 'w') as out_file:

			for json_str in json_list:
				try:
					bl2mp = json.loads(json_str)
					sentence_good = bl2mp["sentence_good"]
					sentence_bad = bl2mp["sentence_bad"]
					sentence_good_reorder = bl2mp["sentence_good_reorder"]
					sentence_bad_reorder = bl2mp["sentence_bad_reorder"]
					bl2mp_type = bl2mp["type"]


					score_good, score_bad, score_good_reorder, score_bad_reorder = mlm_model.sequence_score([sentence_good, sentence_bad, sentence_good_reorder, sentence_bad_reorder], reduction = lambda x: -x.sum(0).item())

					token_good = tokenizer.tokenize(sentence_good)
					token_bad = tokenizer.tokenize(sentence_bad)

					token_good_reorder = tokenizer.tokenize(sentence_good_reorder)
					token_bad_reorder = tokenizer.tokenize(sentence_bad_reorder)


					if verbose:
						print("SENTENCE GOOD")
						print(sentence_good)
						print(score_good)
						print(token_good)
						print("SENTENCE BAD")
						print(sentence_bad)
						print(score_bad)
						print(token_bad)
						print("SENTENCE GOOD REORDER")
						print(sentence_good_reorder)
						print(score_good_reorder)
						print(token_good_reorder)
						print("SENTENCE BAD REORDER")
						print(sentence_bad_reorder)
						print(score_bad_reorder)
						print(token_bad_reorder)

					if len(token_good) == len(token_bad) and len(token_good_reorder) == len(token_good_reorder) and sentence_good != sentence_bad and sentence_good != "" and sentence_bad != "" and sentence_good_reorder != sentence_bad_reorder and sentence_good_reorder != "" and sentence_bad_reorder != "":
						if(score_good < score_bad): #correct for default order
							a+= 1
						t+=1
						if(score_good_reorder < score_bad_reorder): #correct for reorder
							a2+= 1
						t2+=1
						if(score_good_reorder < score_bad_reorder) and (score_good < score_bad): # correct for both orders only
							a3+= 1
						t3+=1

					probs = {"good": score_good, "bad": score_bad}
					probs_reorder = {"good_reorder": score_good_reorder, "bad_reorder": score_bad_reorder}

					tokenized = [[token_good], [token_bad]]
					tokenized_reorder = [[token_good_reorder], [token_bad_reorder]]

					bl2mp['probs'] = probs
					bl2mp['probs_reorder'] = probs_reorder
					bl2mp['tokenized'] = tokenized
					bl2mp['tokenized_reorder'] = tokenized_reorder
					#print((json.dumps(bl2mp) + os.linesep))
					out_file.write(json.dumps(bl2mp) + os.linesep)
				except:
					pass #print("ERROR")
					#print(sentence_good)
					#print(sentence_bad)


	print("ACC:")
	print(f"{a}/{t}")
	print(a/t*100)

	print("REORDER ACC:")
	print(f"{a2}/{t2}")
	print(a2/t2*100)

	print("FREE ORDER ACC:")
	print(f"{a3}/{t3}")
	print(a3/t3*100)

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
