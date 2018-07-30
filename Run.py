
#This file is part of the PAN contest 2018 submission code.
#It is supposed to work with the respective task data for training and testing. 
#Take a look at https://pan.webis.de/clef18/pan18-web/author-identification.html

#to run this software from the command line:
# python Run.py -c folder_with_PAN_files -o folder_with_answers 



import os, torch, codecs, time, random, argparse
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder 
from random import shuffle
from nltk import ngrams
from ModelComp import ArcBinaryClassifier, Encoder, BinaryMeasurer, Parafac
from memory_profiler import profile
from scipy import stats

from collections import Counter, defaultdict
import ModelComp
import json

from sklearn import svm 
from sklearn.decomposition import PCA


global use_cuda
use_cuda = False

ModelComp.use_cuda=use_cuda

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 6})
import matplotlib.pyplot as plt 

import matplotlib.ticker as ticker

from utils import get_corpus_info, get_problem_truth, tagging_problem, create_char_ngrams_stat, \
filter_problem_corpus, create_ngrams_and_splitgrams, stats_for_ngrams_and_skipgrams, \
vectorise_problem_corpus, compute_mean_and_std, make_authors_list, \
define_model, define_optimiser, training, testing, save_tools, load_tools, as_minutes, cluster_test, set_sizes

def main():
    
    
    time_start = time.time()
    
    parser = argparse.ArgumentParser(description='Lukasz Gagala PAN18')
    parser.add_argument('-c', type=str,
                        help='Path to input data set')
    parser.add_argument('-r', type=str,
                        help='input run')
    parser.add_argument('-o', type=str, 
                        help='output dir')
    parser.add_argument('-freq1', type=int, default = 1000,
                        help='first n most frequent items, default: 1000')
    parser.add_argument('-freq2', type=int, default = 300, 
                        help='number of most frequent lemmas added to PoS-tags, default: 300')
    
    args = parser.parse_args()
    problem_names, problem_languages = get_corpus_info(args.c)
    
    for nb, problem in enumerate(problem_names):
        problem_solving(nb, problem, problem_languages, args, time_start)
        
        
        
def problem_solving(nb, problem, problem_languages, args, time_start): 
  

     
    if problem_languages[nb] == "pl":
       pass
   
    print(problem)        
    local_path = get_problem_truth(args.c, problem) 
    print(local_path)
    problem_collection, number_of_texts = tagging_problem(local_path, problem_languages[nb])
   
    print('tagged')
   
    authors = make_authors_list(problem_collection)
    print('authors defined')
   
   
   

       
    freq1 = args.freq1
    freq2 = args.freq2
    
    
    training_set_size, test_set_size = set_sizes(problem_collection)
    
    
    random.seed(time.time())
    
    trunc_words1, trunc_words2 = create_char_ngrams_stat(problem_collection, freq2, problem_languages[nb])

    problem_collection = filter_problem_corpus(problem_collection, trunc_words1, trunc_words2, problem_languages[nb])
    
    problem_collection, nb_categories = create_ngrams_and_splitgrams(problem_collection)
    
    words_encoder, words_num = stats_for_ngrams_and_skipgrams(problem_collection, nb_categories, freq1)

    freq_feature, words_num = vectorise_problem_corpus(problem_collection, words_encoder, words_num, frequency, number_of_texts)
    
    freq_feature_form_norm, network_sizes = compute_mean_and_std(freq_feature, problem_collection,words_num)
    
    
    model_test = define_model(network_sizes, len(authors),len(words_encoder))
    optimiser_test = define_optimiser(model_test)
    bceloss = torch.nn.NLLLoss()
    if use_cuda:
        bceloss = bceloss.cuda()
        
    mseloss = torch.nn.MSELoss()
    if use_cuda:
        mseloss = mseloss.cuda()
        
        
    model = training(model_test, training_set_size, 
                     problem_collection, authors, bceloss, optimiser_test, 
                     freq_feature_form_norm)
    
    print('after training')
    
    result = testing(problem_collection, model, authors, freq_feature_form_norm)
    
    print('after testing')
    
    with open(os.path.join(args.o,'answers-{}.json'.format(problem)), 'w') as outfile:
        json.dump(result, outfile)
    
    time_now = time.time()
    
    timing = time_now -time_start
    print(as_minutes(timing))

  
  
    print('sdadkashdksadfksahfksafhksadhf')    
    return    
if __name__ == '__main__':
    main()