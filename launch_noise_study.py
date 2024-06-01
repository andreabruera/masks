import os

mins = [1, 2, 4, 6, 8, 10]
mins=[4, 6]
mins=[2]
maxs = [1, 2, 4, 6, 8, 10, 12, 14]
maxs=[4, 6]
maxs=[9]
iterations = [1, 10, 50, 100]
iterations=[10]
balances = [True, False]
balances = [False]
models = [
          #'orthography', 'pixelwise', 
          #'wordnet', 
          #'cooc', 'log_cooc',
          #'ppmi', 'w2v', 'fasttext',
          'gpt2', 
          #'bert',
          ]
min_words = [16, 20, 24, 28, 32]
min_words = [8]

for data_split in ['perceptual_awareness', 'objective_accuracy']:
#for data_split in ['perceptual_awareness']:
#for data_split in ['objective_accuracy']:
#for data_split in ['all_cases']:

    for min_word in min_words:
        for model in models:
            for min_erp in mins:
                for max_erp in maxs:
                    if max_erp >= min_erp:
                        for it in iterations:
                            for bal in balances:
                                if bal:
                                    message = 'python3 main.py --analysis time_resolved_rsa --data_split {} --bids_folder /import/cogsci/andrea/dataset/neuroscience/unaware_semantics_bids --computational_model {} --lower_threshold {} --higher_threshold {} --n_iterations {} --minimum_number_words {} --balance_semantic_domains'.format(data_split, model, min_erp, max_erp, it, min_word)
                                else:
                                    message = 'python3 main.py --analysis time_resolved_rsa --data_split {} --bids_folder /import/cogsci/andrea/dataset/neuroscience/unaware_semantics_bids --computational_model {} --lower_threshold {} --higher_threshold {} --n_iterations {} --minimum_number_words {}'.format(data_split, model, min_erp, max_erp, it, min_word)
                                os.system(message)
                                plot_message = '{} --plot'.format(message)
                                os.system(plot_message)
