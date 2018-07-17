
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
use_cuda = False#torch.cuda.is_available()

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

#@profile
def main():
    
    time_start = time.time()
    
    #folder = sys.argv[1]
    parser = argparse.ArgumentParser(description='Lukasz Gagala PAN18')
    parser.add_argument('-c', type=str,
                        help='Path to input data set')
    parser.add_argument('-r', type=str,
                        help='input run')
    parser.add_argument('-o', type=str, 
                        help='output dir')
    args = parser.parse_args()
    problem_names, problem_languages = get_corpus_info(args.c)
    
    #problem_languages = problem_languages*10
    
    for nb, problem in enumerate(problem_names):
        problem_solving(nb, problem, problem_languages, args, time_start)
        
        
        
def problem_solving(nb, problem, problem_languages, args, time_start): 
  

     
    if problem_languages[nb] == "pl":
       pass#continue
   #if (nb != 0 and nb != 0):
   #    continue
   
   
    print(problem)        
    local_path = get_problem_truth(args.c, problem) 
    print(local_path)
    problem_collection, number_of_texts = tagging_problem(local_path, problem_languages[nb])
   
    print('tagged')
   
    authors = make_authors_list(problem_collection)
    print('authors defined')
   
   
   

       
    frequency = 8000#random.choice([500,600,800])
    freq1 = 400#random.choice([100,150,200,250,300,350]) 
    freq2 =  1000#random.choice([100,150,200,250,300,350])
    
    training_set_size, test_set_size = set_sizes(problem_collection)
    
    
    random.seed(time.time())
    
    print(frequency, freq1, freq2)
    #del problem_collection
    trunc_words1, trunc_words2 = create_char_ngrams_stat(problem_collection, freq1, freq2, problem_languages[nb])

    problem_collection = filter_problem_corpus(problem_collection, trunc_words1, trunc_words2, problem_languages[nb])
    
    problem_collection, nb_categories = create_ngrams_and_splitgrams(problem_collection)
    
    words_encoder, words_num = stats_for_ngrams_and_skipgrams(problem_collection, nb_categories, frequency)

    freq_feature, words_num = vectorise_problem_corpus(problem_collection, words_encoder, words_num, frequency, number_of_texts)
    
    freq_feature_form_norm, pca, network_sizes = compute_mean_and_std(freq_feature, problem_collection,words_num)
    
    
    model_test = define_model(network_sizes, len(authors), freq_feature_form_norm,len(words_encoder))
    optimiser_test = define_optimiser(model_test)
    bceloss = torch.nn.NLLLoss()
    if use_cuda:
        bceloss = bceloss.cuda()
        
    mseloss = torch.nn.MSELoss()
    if use_cuda:
        mseloss = mseloss.cuda()
    
    #global model
    model = training([None, model_test], training_set_size, 
                     problem_collection, authors, bceloss, mseloss,(None, optimiser_test), 
                     freq_feature_form_norm, None)
    
    print('after training')
    
    result = testing(problem_collection, model, authors, freq_feature_form_norm, None)
    
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
  
    if True:
        
        #problem = 'problem00001'
        #nb = 0
        if problem_languages[nb] == "pl":
            pass#continue
        #if (nb != 0 and nb != 0):
        #    continue
        
        
        print(problem)        
        local_path = get_problem_truth(args.c, problem) 
        print(local_path)
        #global problem_collection
        problem_collection, number_of_texts = tagging_problem(local_path, problem_languages[nb])
        
        print('tagged')
        
        #gc.collect()
        #save_tools(problem_collection_, 'problem_collection_anfang')
        #save_tools(number_of_texts, 'number_of_texts')
        
        #problem_collection_ = load_tools('problem_collection_anfang')
        #number_of_texts = load_tools('number_of_texts')
        #save_tools(number_of_texts, 'number_of_texts')
        authors = make_authors_list(problem_collection)
        print('authors defined')
        
        #quit()
        results = []
        
        #frequency = random.choice([200,220,240,260,280,300])
        #freq1 = random.choice([100,150,200,250,300,350]) 
        #freq2 =  random.choice([100,150,200,250,300,350])
        #frequency_ = [200,220,240,260,280,300]
        #freq1_ = [100,150,200,250,300,350]
        #freq2_ =  [100,150,200,250,300,350]
        
        
        
        if True:#for x in range(1):
            #break
            #problem_collection = copy.deepcopy(problem_collection_)
            
            #frequency = 3000#random.choice([500,600,800,1000,1200,1500])
            #freq1 = 100#random.choice([100,150,200,250,300,350]) 
            #freq2 =  200#random.choice([100,150,200,250,300,350])
            
            #training_set_size, test_set_size = set_sizes(problem_collection)
            
            
            #random.seed(time.time())
            
            #print(frequency, freq1, freq2)
            #trunc_words1, trunc_words2 = create_char_ngrams_stat(problem_collection, freq1, freq2, problem_languages[nb])
        
            #problem_collection = filter_problem_corpus(problem_collection, trunc_words1, trunc_words2, problem_languages[nb])
            
            #problem_collection, nb_categories = create_ngrams_and_splitgrams(problem_collection)
            
            #words_encoder, words_num = stats_for_ngrams_and_skipgrams(problem_collection, nb_categories, frequency)
     
            #problem_collection, freq_feature, words_num = vectorise_problem_corpus(problem_collection, words_encoder, words_num, frequency, number_of_texts)
            
            #freq_feature_form_norm, pca, network_sizes = compute_mean_and_std(freq_feature, problem_collection,words_num)
            ################################
            #noisy_labels = cluster_test(problem_collection, len(freq_feature), authors, freq_feature_form_norm)
            
            
            
            frequency = 8000#random.choice([500,600,800])
            freq1 = 400#random.choice([100,150,200,250,300,350]) 
            freq2 =  1000#random.choice([100,150,200,250,300,350])
            
            training_set_size, test_set_size = set_sizes(problem_collection)
            
            
            random.seed(time.time())
            
            print(frequency, freq1, freq2)
            #del problem_collection
            trunc_words1, trunc_words2 = create_char_ngrams_stat(problem_collection, freq1, freq2, problem_languages[nb])
        
            problem_collection = filter_problem_corpus(problem_collection, trunc_words1, trunc_words2, problem_languages[nb])
            
            problem_collection, nb_categories = create_ngrams_and_splitgrams(problem_collection)
            
            words_encoder, words_num = stats_for_ngrams_and_skipgrams(problem_collection, nb_categories, frequency)
     
            freq_feature, words_num = vectorise_problem_corpus(problem_collection, words_encoder, words_num, frequency, number_of_texts)
            
            freq_feature_form_norm, pca, network_sizes = compute_mean_and_std(freq_feature, problem_collection,words_num)
            
            
            #result = cluster_test(problem_collection, len(freq_feature), authors, freq_feature_form_norm)
            
            
            
            
            #save_tools(problem_collection, 'problem_collection')
            #save_tools(words_encoder, 'words_encoder')
            #save_tools(words_num, 'words_num')
            #save_tools(freq_feature, 'freq_feature')
            
            #problem_collection = load_tools('problem_collection')
            #words_encoder = load_tools('words_encoder')
            #words_num = load_tools('words_num')
            #freq_feature = load_tools('freq_feature')
            #print('tutaj')
            
            
            
            
            #global model_test
            #model_train = define_model(network_sizes, len(authors), freq_feature_form_norm,len(words_encoder))
            model_test = define_model(network_sizes, len(authors), freq_feature_form_norm,len(words_encoder))
            #model = define_model(network_sizes, len(authors), freq_feature_form_norm,len(words_encoder))
            
            #global optimiser_test 
            
            #optimiser_train = define_optimiser(model_train)
            optimiser_test = define_optimiser(model_test)
            bceloss = torch.nn.NLLLoss()
            if use_cuda:
                bceloss = bceloss.cuda()
                
            mseloss = torch.nn.MSELoss()
            if use_cuda:
                mseloss = mseloss.cuda()
            
            #global model
            model = training([None, model_test], training_set_size, 
                             problem_collection, authors, bceloss, mseloss,(None, optimiser_test), 
                             freq_feature_form_norm, None)
            
            print('after training')
            
            result = testing(problem_collection, model, authors, freq_feature_form_norm, None)
            
            print('after testing')
            
            with open(os.path.join(args.o,'answers-{}.json'.format(problem)), 'w') as outfile:
                json.dump(result, outfile)
            
            #results.append(result)
            
            del model_test, optimiser_test, bceloss, mseloss, outfile
            #gc.collect()
            del freq_feature_form_norm, pca, network_sizes, result, freq_feature, words_num
            #gc.collect()
            del trunc_words1, trunc_words2, nb_categories, words_encoder, training_set_size, test_set_size
            #gc.collect()
            del problem_collection,model
            #del globals()['problem_collection'], globals()['model']
            #del globals()['optimiser_test']
            #del globals()['model_test']
            #gc.collect()
            time_now = time.time()
            
            timing = time_now -time_start
            print(as_minutes(timing))
        
            #gc.collect()
        
        del number_of_texts, authors
        gc.collect()
        
        #save_tools(results, problem)
        #quit()
       
        #quit()
    
    
    
    print('sdadkashdksadfksahfksafhksadhf')    
    return    
if __name__ == '__main__':
    main()
