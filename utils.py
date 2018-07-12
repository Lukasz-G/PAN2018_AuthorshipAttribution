import os, torch, gc, codecs, time, math, random, numpy, heapq, configparser, sys, pickle, itertools#, morfeusz2
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from random import shuffle
from nltk import ngrams
from ModelComp import ArcBinaryClassifier, Encoder, StyloMeasurer, Parafac
from memory_profiler import profile
from sklearn.multiclass import OneVsRestClassifier


from scipy import stats

from collections import Counter, defaultdict
import ModelComp
import json

#from sklearn import svm
#from sklearn.decomposition import PCA
#from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
#from sklearn.neighbors import KNeighborsClassifier
#from nltk.parse.corenlp import CoreNLPDependencyParser
#from Stylus1.Styl1 import model



global use_cuda
use_cuda = False#torch.cuda.is_available()

ModelComp.use_cuda=use_cuda
Config = configparser.ConfigParser()

#import matplotlib
#matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 6})
#import matplotlib.pyplot as plt 

import matplotlib.ticker as ticker


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+0.01)
    rs = es - s
    return '%s ( %s)' % (as_minutes(s), as_minutes(rs))


'''a function for saving prepared data for training'''
def save_tools(tool, name):
    with open('{}.p'.format(name), 'wb') as f:
        print('fragment file format: {}'.format(name))
        pickle.dump(tool, f)
    return

''' a function for loading data'''
def load_tools(name):
    name = name + '.p'
    with open(name, 'rb') as data:
        tool = pickle.load(data)
        return tool

def identity(document):
    return document


def index_words(words):
    
    words_encoder = LabelEncoder()
    words_encoder.fit(words)# + ['<UNK>'])
    return words_encoder

def leave_only_alphanumeric(string):
    return ''.join(ch if ch.isalnum() else ' ' for ch in string)

def repackage_variable(h):
    if type(h) == Variable:
        if use_cuda:
            return Variable(h.data).cuda()
        else:
            return Variable(h.data)
    
    else:
        return tuple(repackage_variable(v) for v in h)


def vectorising_words(words, words_encoder):
    
    words_vectorised = []
    
    for word in words:
        try:
            word_vectorised = words_encoder.transform([word])
            word = torch.from_numpy(word_vectorised)
        
        except:
            word_vectorised = words_encoder.transform(['<UNK>'])
            word = torch.from_numpy(word_vectorised)
            
        words_vectorised.append(word)
    
    words_vectorised = torch.cat(words_vectorised,0)
        
    seq = torch.unsqueeze(words_vectorised,1).long()
    
    return seq

def vectorising_words_2(words, words_encoder):
    
    words_vectorised = []
    
    for word in words:
        try:
            word_vectorised = words_encoder.transform([word])
            word = word_vectorised.tolist()#torch.from_numpy(word_vectorised)
        
        except:
            word_vectorised = words_encoder.transform(['<UNK>'])
            word = word_vectorised.tolist()#torch.from_numpy(word_vectorised)
            
        words_vectorised.append(word)
    
    #words_vectorised = torch.cat(words_vectorised,0)
        
    #seq = torch.unsqueeze(words_vectorised,1).long()
    
    return words_vectorised#seq






'''two auxiliary function for reading data of a parameters file'''
def isInt(inputString):
    return all(char.isdigit() for char in inputString)
########################################################
def isFloat(inputString):
    return all(char.isdigit() or char == '.' for char in inputString)
########################################################



  
'''a function for creating a dictionary of parameters'''
def get_param_dict(p):
    config = configparser.configparser()
    config.read(p)
    # parse the param
    param_dict = dict()
    for section in config.sections():
        for name, value in config.items(section):
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif isInt(value):
                value = int(value)
            elif isFloat(value):
                value = float(value)
            param_dict[name] = value
    return param_dict



def genarator_samples(text, seq_length):
    
    num = 0
    
    for z in range(len(text)-seq_length):
        seq = [[],[]]
        if z % seq_length == 0:
    
            #z = z-1
            #if z <= 0:
            #    z = 0
            #print(z,'dfhdxdxvcvbcxmvxcxcmbvxcm')    
            
            for x in range(seq_length):
                y = z + x
                #print(y,'gdfgdfgdgfdgdf')
                try:
                    seq[0].append(text[y])
                    seq[1].append(text[y+1])
                    #print(y,y+1,'dfdvccvbghhfg')
                except:
                    continue
            
            
            num+=1
            yield seq,num



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, output1, output2, label):
        
        
        #max_len = min(output1.size()[0], output2.size()[0])
        
        #print(output1, output2)
        #output1, output2 = output1[:max_len], output2[:max_len] 
        #quit()
        #output1, output2 = self.softmax(output1.view(-1)), self.softmax(output2.view(-1))
        #output1, output2 = output1.view(-1).unsqueeze(0), output2.view(-1).unsqueeze(0)
        
        #output1, output2 = self.softmax(output1), self.softmax(output2)
        #quit()
        manhattan_distance = torch.nn.functional.pairwise_distance(output1.view(-1).unsqueeze(0), output2.view(-1).unsqueeze(0),p=1)
        #manhattan_distance = 2*(torch.sigmoid(manhattan_distance)-0.5)
        #print(euclidean_distance)
        
        
        
        #quit()
        #euc_dist = torch.exp(-(torch.norm(output1.squeeze()-output2.squeeze(),p=1)))
        
        #loss_contrastive = torch.nn.functional.mse_loss(manhattan_distance,label)
        
        
        #print(manhattan_distance,'sdgdfgdfgfd')
        #quit()
        loss_contrastive = torch.mean((1-label) * manhattan_distance +
                                      (label) * torch.clamp(self.margin - manhattan_distance, min=0.0))

        #loss_contrastive = torch.nn.functional.mse_loss(euclidean_distance,label)
        #loss_cos_con = torch.mean((1-label) * 1.0-euclidean_distance +
        #                            (label) * euclidean_distance * torch.lt(euclidean_distance, self.margin).float())
        
        
        
        
        #euclidean_distance = 2*(torch.sigmoid(euclidean_distance) - 0.5) 
        
        
        return loss_contrastive, manhattan_distance



def generatorek(texts):
    length = len(texts)
    for x in range(length):
        for y in range(length-x):
            if x+y+1 == length:
                break
            yield x,(texts[x],texts[y+x+1])

def training_comb1(list,x):
    for nb, elem1 in enumerate(list):
        yield elem1,elem1,x

def training_comb2(list):
    for nb, elem1 in enumerate(list):
        for elem2 in list[nb:]:
            pipo = torch.sum(torch.eq(elem1,elem2))/elem1.size()[0]
            #print(pipo)
            #quit()
            if pipo == 1.0:
                pass
            else:
                yield elem1,elem2

def training_comb3(list1,list2):
    for elem1 in list1:
        for elem2 in list2:
                yield elem1,elem2


def new_generator_texts(problem):
    nb_text = -1
    same_authors = []
    nb_same_authors = 0
    list_= [[author, texts] for author, texts in problem.items()]
    random.seed(time.time())
    random.shuffle(list_)
    list_of_works = []
    for author, texts in list_:
        
        #print(author, texts)
        #quit()
        same_author = []
        
        for text, text_self in texts.items():
            nb_text += 1
            
                #quit()
                #same_author.append(text_self)
            #if author != 'unknown':
                #print(nb_text,'nb_text')
            list_of_works.append([author, text_self, nb_text])
                #nb_same_authors += 1
            
            #with unknown text
            #unknown_texts = []
            #for pair in truths_for_problem["ground_truth"]:
            #    if pair['true-author']==author:
            #        unknown_texts.append(problem['unknown'][pair['unknown-text']][0])
            #print(len(unknown_texts),'sfdsfdsfds')
                  
            #same_authors.append(same_author)
    
    #random.
    
    random.seed(time.time())
    random.shuffle(list_of_works)
    
    for x in list_of_works:
        yield x
    
    

def generator_texts(problem, truths_for_problem, rank, size):
    
    
    
    same_authors = []
    nb_same_authors = 0
    for author, texts in problem.items():
        same_author = []
        if author != 'unknown':
            for text, text_self in texts.items():
                #print('sdfsdfsdf',text_self)
                #quit()
                same_author.append(text_self)
                nb_same_authors += 1
            
            #with unknown text
            #unknown_texts = []
            #for pair in truths_for_problem["ground_truth"]:
            #    if pair['true-author']==author:
            #        unknown_texts.append(problem['unknown'][pair['unknown-text']][0])
            #print(len(unknown_texts),'sfdsfdsfds')
                  
            same_authors.append(same_author)
    
    
    
    different_authors = []
    written_by_same_author = []
    for nb1, author1 in enumerate(same_authors):
        for nb2, author2 in enumerate(same_authors): 
            if nb1 >= nb2:
                if nb1 > nb2:
                    #print(nb1,nb2)
                    putptu = [x for x in training_comb3(author1, author2)]
                    #print(len(putptu),'ghghghghghghg')
                    different_authors.extend([x for x in training_comb3(author1, author2)])
                if nb1 == nb2:
                    written_by_same_author.extend([x for x in training_comb2(author2)])
    
    
    
    
    #random.seed(time.time())
    
    negative_examples = []
    
    length = len(written_by_same_author)
    #print(length,'gfdgdgdgdf')
    corpus_part = length/size 
    
    
    #written_by_same_author_part = written_by_same_author[(corpus_part):(corpus_part+corpus_part)]
    
    
    for w in range(len(written_by_same_author)):
        random.seed(time.time()+w)
        random_pair = random.randint(0,len(different_authors)-1)
        #print(random_pair)
        negative_examples.append(different_authors[random_pair])
    
    
    
 
    #print(len(written_by_same_author[0]),'dfgdfgdgdgdfg')
    
    number = 0
    for x in range(100):#(len(written_by_same_author_part)-1)*2):
        number += 1
        if x % 2 ==0:
            yield number, written_by_same_author[x//1][0], written_by_same_author[x//1][1], 'Y'
        else:
            yield number, negative_examples[x//2][0], negative_examples[x//2][1],'N'
  

def generator_evaluation(problem):
    
    for author, texts in problem.items():
        if author != 'unknown':
            for _, text_self in texts.items():
                yield author, text_self
 






def save_plot(points, folder=None, plot_name=None, base=1):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=base)
    ax.yaxis.set_major_locator(loc)
    #print(len(points))
    plt.plot(points)
    plt.savefig(os.path.join(folder, plot_name))


def save_plot2(points, label=None,folder=None, plot_name=None, base=1):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=base)
    ax.yaxis.set_major_locator(loc)
    
    points1 = (points[:,0])
    #points1_ = numpy.array(points[:,2]-points[:,1])
    points2 = (points[:,1])
    points3 = (points[:,2])
    points4 = (points[:,3])
    #print(points1, points2,'fdsdvdvvcvcvcv')
    
    #quit()
    
    ax.scatter(points1,points2)
    ax.scatter(points1,points3)
    ax.scatter(points1,points4)
    
    #for x,y,label in zip(points[:,0],points[:,1], labels):
    #    ax.text(x,y,label)
    
    #print(len(points))
    plt.savefig(os.path.join(folder, plot_name))


def save_plot3(points, labels=None,folder=None, plot_name=None, truth=None):
    
    #points_ = zip[points]
    #truth["ground_truth"]
    #['true-author']==best_candidate and pair['unknown-text']
    #base = round(max(max(points_[0]), max(points_[1]))/10,1)
    #zipped_points = zip(points)
    #print(zipped_points[0])
    #zipped_points_setted = set(zipped_points[0])
    #all_tested_names =list(zipped_points_setted)
    
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=base)
    #ax.yaxis.set_major_locator(loc)
    #all_results = []
    
    n = 0
    for_x_lables = []
    for unknown_texts, candidate_lists in points.items():
        #text, candidate_names, candidate_points, candidate_points_averaged, candidate_names_averaged = p
        for pair in truth["ground_truth"]:#[n]['true-author']
            if pair['unknown-text']==unknown_texts:
                true_author = pair['true-author']
        for_x_lables.append(unknown_texts)
        
        
        #for num, candidate_result in enumerate(candidate_points):
            #position = all_tested_names.index(text_name)
            #print(text,candidate_result)
        #    if candidate_names[num] == true_author:
        #        ax.scatter(candidate_result, n, marker='*')
        #        ax.text(candidate_result,n,candidate_names[num])
        #    else:
        #        ax.scatter(candidate_result, n, alpha=0.05, s=20*4**2)
        #    
        #    all_results.append(candidate_result)
        for candidate, candidate_results in candidate_lists.items():
            if candidate == true_author:
                #for r in candidate_results:
                #    ax.scatter(r, n, marker='*')
                #    ax.text(r,n,candidate)
                
                ax.scatter(sum(candidate_results)/len(candidate_results), n, alpha=0.5,marker='8', s=20*4**2)
                #ax.text(candidate_averaged_result,n,candidate_names_averaged[num])
            else:
                #for r in candidate_results:
                #    ax.scatter(r, n, alpha=0.15, marker='p', s=20*4**2)
                ax.scatter(sum(candidate_results)/len(candidate_results), n, alpha=0.5,marker='*', s=20*4**2)
            #all_results.append(candidate_averaged_result)
        
        n += 1
    
    #for x,y,label in zip(points[:,0],points[:,1], labels):
    #    ax.text(x,y,label)
    
    plt.yticks(range(len(for_x_lables)), for_x_lables)
    
    #plt.xlim(min(all_results), max(all_results))
    #print(len(points))
    plt.savefig(os.path.join(folder, plot_name))

def save_plot4(data, folder=None, plot_name=None):
    
  
    plt.figure()
    fig, ax = plt.subplots()
 
    for d in data:
        truth, candidate_result = d
        
        if truth == 'Y':
            ax.scatter(1,candidate_result)
            #ax.annotate(candidate_names[nb], (num,x)) 
        if truth == 'N':
            ax.scatter(0,candidate_result)
        
        #for nb, x in enumerate(candidate_results):
        #    ax.scatter(num,x)
        #    ax.annotate(candidate_names[nb], (num,x)) 
    
 
    plt.savefig(os.path.join(folder, plot_name))

def compute_cross_entropy(log_prob, target):
    # compute reconstruction loss using cross entropy
    loss = [torch.nn.functional.nll_loss(sentence_emb_matrix, word_ids, size_average=False) for sentence_emb_matrix, word_ids in zip(log_prob, target)]
    average_loss = sum([torch.sum(l) for l in loss]) / log_prob.size()[0]
    return average_loss





def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def encoder(model=None, optimizer=None,frag_to_encode=None):
    frag_to_encode=Variable(frag_to_encode)
    if use_cuda:
        frag_to_encode=frag_to_encode.cuda()
    model.train()
    model.zero_grad()
    embbeded_frag, compressed_frag, reconstructed_frag = model(frag_to_encode)
    
    frag_to_encode = repackage_variable(frag_to_encode)
    
    #print(frag_to_encode.view(-1), reconstructed_frag, 'fsdhfkdshkfdskfghkjds')
    #quit()
    loss = torch.nn.functional.nll_loss(reconstructed_frag.squeeze(), frag_to_encode.view(-1))
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-1.0,1.0)
    
    
    optimizer.step()
    results = reconstructed_frag.squeeze().data.cpu().topk(1,dim=1)[1].view(-1)#.numpy()           
    #print(frag_to_encode.view(-1).data.cuda(),results,'gfdggvcbcvbvbcwrwerw')
    comp_ = torch.eq(frag_to_encode.view(-1).data.cpu(),results)            
    acc = float(torch.sum(comp_))/float(torch.numel(comp_))
    
    #print(acc,'acc')
    
    
    return compressed_frag, loss.data[0], acc

def encoder_integrated(model=None, frag_to_encode=None):
    
    
    
    #frag_to_encode=Variable(frag_to_encode)
    #if use_cuda:
    #    frag_to_encode=frag_to_encode.cuda()
    model.train()
    #model.zero_grad()
    embbeded_frag, compressed_frag, reconstructed_frag = model(frag_to_encode)
    
    #frag_to_encode = repackage_variable(frag_to_encode)
    
    #print(compressed_frag.size(),'fsdhfkdshkfdskfghkjds')
    #quit()
    #loss = torch.nn.functional.nll_loss(reconstructed_frag.squeeze(), frag_to_encode.view(-1))
    
    #results = reconstructed_frag.squeeze().data.cpu().topk(1,dim=1)[1].view(-1)#.numpy()           
    #print(frag_to_encode.view(-1).data.cuda(),results,'gfdggvcbcvbvbcwrwerw')
    #comp_ = torch.eq(frag_to_encode.view(-1).data.cpu(),results)            
    #acc = float(torch.sum(comp_))/float(torch.numel(comp_))
    
    #print(acc,'acc')
    
    
    return compressed_frag, None, None#, loss, acc



def encoder_eval(model=None,frag_to_encode=None):
    
    #frag_to_encode=Variable(frag_to_encode, volatile=True)
    if use_cuda:
        frag_to_encode=frag_to_encode.cuda()
    
    model.eval()
    
    embbeded_frag, compressed_frag, reconstructed_frag = model(frag_to_encode)
    
    #frag_to_encode = repackage_variable(frag_to_encode)
    
    #print(frag_to_encode.view(-1), reconstructed_frag, 'fsdhfkdshkfdskfghkjds')
    #quit()
    #loss = torch.nn.functional.nll_loss(reconstructed_frag.squeeze(), frag_to_encode.view(-1))
    #results = reconstructed_frag.squeeze().data.cpu().topk(1,dim=1)[1].view(-1)#.numpy()           
    #print(frag_to_encode.view(-1).data.cuda(),results,'gfdggvcbcvbvbcwrwerw')
    #comp_ = torch.eq(frag_to_encode.view(-1).data.cpu(),results)            
    #acc = float(torch.sum(comp_))/float(torch.numel(comp_))
    
    
    
    return compressed_frag, None, None#loss.data[0], acc


def sigmoid_annealing_schedule(step, max_step, param_init=1.0, param_final=0.01, gain=0.2):
    return ((param_init - param_final) / (1 + math.exp(gain * (step - (max_step / 2))))) + param_final



def skipgram(text,skipgram_number):
    
    text_ = []
    for ngram_ in strange(ngrams(text, skipgram_number)):
    
        start = ngram_[0]
        end = ngram_[-1]
        empty = ['EMPTY']# for x in range(skipgram_number-2)]
        name_skipgram = '-'.join([start]+empty+[end])
        #print(type(name_skipgram),name_skipgram)
        #quit()
        text_.append(name_skipgram)
        del name_skipgram, ngram_, start, end, empty
    #print(text_)
    #quit()
    return text_
        
        
        

def minmax(x, y):
    mins, maxs = 0.0, 0.0
    for i in range(len(x)):
        a, b = x[i], y[i]
        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a
    if maxs > 0.0:
        return 1.0 - (mins / maxs)
    return 0.0


def dynamic_split(tensor, length, shift_size, words_encoder):
        
        #print(tensor.size())
        #quit()
        
        
        shift = 0
        length_= length
        yielding = True
        while yielding:
            if tensor.size()[0] > length_+shift:
                new_tensor = tensor[shift:length_+shift]
                shift += shift_size
                yield new_tensor
            
            elif tensor.size()[0] <= length_+shift:
                new_tensor = tensor[shift:]
                diff = length - new_tensor.size()[0]
                pad = ['<PAD>' for _ in range(diff)]
                #print(new_tensor.size()[0], length, diff)
                if pad:
                    pad = vectorising_words(pad, words_encoder)
                    new_tensor = torch.cat([new_tensor,pad], dim=0)
                yielding = False
                yield new_tensor

def thousand_split(tensor, words_encoder):
        
        frag_size = 1000
        
        if tensor.size()[0] >= frag_size:
            start_range = tensor.size()[0] - frag_size
            random.seed(time.time())
            start_point = random.choice(range(start_range))
            return tensor[start_point:start_point+frag_size]
        elif tensor.size()[0] < frag_size:
            diff = frag_size - (tensor.size()[0])
            half_diff = diff//2
            #return tensor
        #################################
            pad = ['<PAD>' for _ in range(diff)]
            if pad:
                pad = vectorising_words(pad, words_encoder)
                if half_diff:
                    new_tensor = torch.cat([pad[:half_diff],tensor,pad[half_diff:]], dim=0)
                else:
                    new_tensor = torch.cat([tensor,pad], dim=0)
                return new_tensor
            

def get_corpus_info(folder):
    
    data_on_collection = r"collection-info.json"
    description_collection = json.load(open(os.path.join(folder,data_on_collection)))
    problem_names = [x['problem-name'] for x in description_collection]
    problem_languages = [x['language'] for x in description_collection]
    
    return problem_names, problem_languages

def get_problem_truth(folder_address,problem_name):
    
    local_path = os.path.join(folder_address,problem_name)
    #truths_for_problem = json.load(open(os.path.join(local_path,'ground-truth.json')))
    return local_path#, truths_for_problem


def Polish_tagging(text):
    
    
    
    #print(text)
    text_ = ' '.join(text)
    
    #doc = morfeusz2.Morfeusz().analyse(text_)
                #doc = nlp(t.read())
                
    number_words_in_text = []
    tagged_text = []
    for d in morfeusz2.Morfeusz().analyse(text_):
        #no_word, nb_option, description = d
        #print(d)         
        if not(d[0] in number_words_in_text):
            number_words_in_text.append(d[0])
            #print(description)
            #print(text[no_word])
            tagged_text.append([d[2][0], d[2][2].split(':')[0]])
    
    del d, text_, number_words_in_text
    #gc.collect()
    
    return tagged_text, text

def strange(gen):
    for thing in gen:
        yield thing

#@profile
def tagging_problem(path_to_problem, language):
    
    #global problem_collection
    
    problem_collection = {}


    if language == 'sp':
        language = 'es'
    
    if language == 'pl':
        #morfeusz2 = None
        #nlp = morfeusz2
        import morfeusz2
        nlp = reload(morfeusz2).Morfeusz()
        pass#nlp = morfeusz2#.Morfeusz
    else:
        print('before loading')
        #nlp = spacy
        import spacy
        nlp = reload(spacy).load(language)
        
    print('nlp loaded')

    number_of_texts = 0
    for dirname in os.listdir(path_to_problem):
        
        #print(dirname, language)
        ##################
        #break
        
        
        author={}
        
        
        
        
        
        
        if os.path.isdir(os.path.join(path_to_problem,dirname)):
            
            
            
            for file in os.listdir(os.path.join(path_to_problem,dirname)):
                number_of_texts += 1
                #print(dirname, file,language)
                text_entry = {}
                text = []
                t = codecs.open(os.path.join(path_to_problem,dirname,file), 'r', 'utf-8')
                #t = open(os.path.join(path_to_problem,dirname,file))
                for word in strange(t.read().split()):
                    #word = leave_only_alphanumeric(word)
                    #word = word.lower().strip()
                    text.append(leave_only_alphanumeric(word).lower().strip())
                    del word
                #text_ = ' '.join(text)#.decode('utf-8')
                t.close()
                del t
                if language != 'pl':
                    
                    #doc = nlp(text_)
                    text_ready = []
                    for token in strange(nlp(' '.join(text))):
                        if token.pos_ != 'PUNCT':
                        #if token.lemma_:
                            #if token.pos_ in ('CONJ','DET','ADP','PART'):
                            if language=='en':
                                #tag_ = token.tag_.split('_')[0]
                            
                                text_ready.append([token.text, token.lemma_,token.tag_])
                            
                            if language=='fr':
                                tag_ = token.tag_.split('_')[0]
                            
                                text_ready.append([token.text, token.lemma_,tag_])
                                del tag_
                            if language=='it':
                                tag_ = token.tag_.split('_')[0]
                                
                                text_ready.append([token.text, token.lemma_,tag_])
                                del tag_
                            if language=='es':
                                tag_ = token.tag_.split('_')[0]
                                text_ready.append([token.text, token.lemma_,tag_])
                                del tag_
                        del token       
                    #del text_ 
                            #print([token.text, token.lemma_,tag_])
                    del text  
                elif language == 'pl':
                    
                    #text_ = ' '.join(text)
                    
                    number_words_in_text = []
                    tagged_text = []
                    
                    for d in strange(nlp.analyse(' '.join(text))):      
                        if not(d[0] in number_words_in_text):
                            number_words_in_text.append(d[0])
                            tagged_text.append([d[2][0], d[2][2].split(':')[0]])
                        del d
                    #del morfeusz2.Morfeusz().analyse(text_)
                    del number_words_in_text
                    
                    #text_ready = Polish_tagging(text)
                        #else:
                        #    text_ready.append(['<UNK>',token.tag_])
                        #text_ready.append([token.tag_])#token.lemma_, 
                    text_ready = [tagged_text, text]
                    del tagged_text, text
                    #print(text_ready[0][])
                #text_entry[file] = text_ready
                author[file] = text_ready
                del text_ready
        
            problem_collection[dirname] = author
            del author
    #quit()            
    #del nlp
    if language == 'pl':
        pass#del morfeusz2#nlp = reload(morfeusz2)
    else:
        pass#del spacy#nlp = reload(spacy)
    
    del nlp
    gc.collect()
    
    return problem_collection, number_of_texts



def create_char_ngrams_stat(problem_collection, freq1, freq2, language):
    
    flattened_all_text_ngrams = []
    flattened_all_text_lemma = []
    if language != 'pl':
        for k1,v1 in strange(problem_collection.items()):
            for k2,v2 in strange(v1.items()):
                #print(k1,k2)
                for word in v2:
                    #if type(word) == float:
                    #print(word)
                    w = '_' + word[0] + '_'
                    #w_ngrmed = ngrams(w,3)
                    #print([x for x in w_ngrmed])
                    #quit()
                    w_ngrmed = ['-'.join(list(tri)) for tri in strange(ngrams(w,3))]
                    flattened_all_text_ngrams.extend(w_ngrmed)
                    flattened_all_text_lemma.extend([word[1]])
                    del word, w_ngrmed, w
                del k2,v2
            del k1,v1
                    
                    #print(flattened_all_text, word[0])
                    #quit()
    elif language == 'pl':
        for k1,v1 in strange(problem_collection.items()):
            for k2,v2 in strange(v1.items()):
                tagged, raw = v2
                for word in strange(raw):
                    w = '_' + word + '_'
                    #w_ngrmed = ngrams(w,3)
                    #print([x for x in w_ngrmed])
                    #quit()
                    w_ngrmed = ['-'.join(list(tri)) for tri in strange(ngrams(w,3))]
                    flattened_all_text_ngrams.extend(w_ngrmed)
                    del word, w_ngrmed, w
                for tag in strange(tagged):
                    flattened_all_text_lemma.extend([tag[0]])
                    del tag
                del k2,v2, tagged, raw
            del k1,v1
               
    #cnt1 = Counter(flattened_all_text_ngrams)
    trunc_words1 = [k for k, _ in strange(Counter(flattened_all_text_ngrams).most_common())]
    #cnt2 = Counter(flattened_all_text_lemma)
    trunc_words2 = [k for k, _ in strange(Counter(flattened_all_text_lemma).most_common(freq2))]
    
    del flattened_all_text_ngrams, flattened_all_text_lemma, 
    problem_collection
    
    
    #gc.collect()
    
    return trunc_words1, trunc_words2
    

    
def filter_problem_corpus(problem_, trunc_words1, trunc_words2, language): 
    
    #global problem_collection
    
    problem_collection = problem_
    
    for author, texts in strange(problem_collection.items()):
        for text_name, text_self in strange(texts.items()):
            text = [[],[],[],[],[]]
            
            if language != 'pl':
            
                for a in strange(text_self):
                    token = a[0]
                    lemma = a[1]
                    w = '_' + token + '_'
                    #print(w)
                    #quit()
                    #token_ = ngrams(w,3)
                    #l_ngrmed = ['-'.join(list(tri)) for tri in strange(ngrams(w,3))]
                    #lemma = [lemma]
                    word_frags = []
                    for tri in strange(ngrams(w,3)):
                    #for l in l_ngrmed:
                        tri = '-'.join(list(tri))
                        if tri in trunc_words1:
                            word_frags.append(tri)
                        del tri 
                            #letters_=[lemma]#list(lemma)#[lemma]#list(lemma)
                    #del l_ngrmed
                    #else:
                    #    letters_=[]
                    #letters_ = [] if letters_ == ['<UNK>'] else letters_
                    x = word_frags# + [a[1]]
                    if lemma in trunc_words2:
                        y = [lemma] + [a[2]]
                        k = [lemma]
                    else:
                        #pass
                        y = [a[2]]
                        k = [lemma]  
                    
                    z = [a[1]] + [a[2]]
                    
                    
                       
                    text[0].extend(x)
                    text[1].extend(y)
                    text[2].extend(k)
                    text[3].extend(z)
                    text[4].extend([a[2]])
                #print(text)
                #quit()
            
            
            
            elif language == 'pl':
                
                for a in text_self[0]:
                    lemma = a[0]
                    pos = a[1]
       
                    if lemma in trunc_words2:
                        y = [lemma] + [pos]
                        k = [lemma]
                    else:
                        #pass
                        y = [pos]
                        k = [lemma]  
                    
                 
                    
                       
                    text[1].extend(y)
                    text[2].extend(k)
                    text[4].extend([pos])
                for a in text_self[1]:
                    w = '_' + a + '_'
                    #print(w)
                    #quit()
                    #token_ = ngrams(w,3)
                    #l_ngrmed = ['-'.join(list(tri)) for tri in strange(ngrams(w,3))]
                    #lemma = [lemma]
                    word_frags = []
                    for tri in strange(ngrams(w,3)):
                    #for l in l_ngrmed:
                        tri = '-'.join(list(tri))
                        if tri in trunc_words1:
                            word_frags.append(tri)
                        del tri
                    x = word_frags
                    
                    text[0].extend(x)
                    
                
                   
                
                
            del text_self
                
            problem_collection[author][text_name] = text
            del text
        del texts, author
    del trunc_words1, trunc_words2, problem_
    #gc.collect()
    
    return problem_collection

def create_ngrams_and_splitgrams(problem_, unigrams_char= True, unigrams_lemma= True):
    
    #global problem_collection
    problem_collection = problem_
    
    for author, texts in strange(problem_collection.items()):
        for text_name, text_self in strange(texts.items()):
            
            #print(len(text_self))
            #quit()
            
            #tetragrams = ngrams(text_self[0], 4)
            #skip3 = skipgram(text_self[0], 3)
            #trigrams = ngrams(text_self[1], 3)
            #trigrams2 = ngrams(text_self[4], 3)
            #tetragrams = ngrams(text_self[0], 4)
            
            
            
            #if text_name == 'known00005.txt' and author == 'candidate00004':
            
            #    print(text_name, author,text_self[1], text_self[2])
            #    quit()
          
            
            #skip4 = skipgram(text_self[0], 5)
            #unigrams = ngrams(text_self[2], 1)
            
            
            skip_1 = skipgram(text_self[1], 3)
            skip_2 = skipgram(text_self[1], 4)
            skip_3 = skipgram(text_self[1], 5)
            #skip_4 = skipgram(text_self[1], 6)
            #skip_5 = skipgram(text_self[1], 7)
            #skip_com1 = skip_1 + skip_2  
            #skip_3 + skip_4 + skip_5 
            #skip_com2 = skip_3 + skip_4 + skip_5 
            #skip_1 = skipgram(text_self[1], 3)
            #skip_2 = skipgram(text_self[1], 4)
            #skip_3 = skipgram(text_self[1], 5)
            #skip_4 = skipgram(text_self[1], 6)
            #skip_5 = skipgram(text_self[1], 7)
            #skip_pos = skip_1 + skip_2 + skip_3 + skip_4 + skip_5 
            
            
            
            
            
            
            
            
            #skip4 = skipgram(text_self[0], 5)
            #unigrams_char = ngrams(text_self[0], 1)
            #unigrams_lemma = ngrams(text_self[2], 1)
            
            #tetragrams = ['-'.join(list(bi)) for bi in tetragrams]
            
            trigrams = ['-'.join(list(tri)) for tri in strange(ngrams(text_self[1], 3))]
            trigrams2 = ['-'.join(list(tri)) for tri in strange(ngrams(text_self[4], 3))]
            unigrams_char = ['-'.join(list(uni)) for uni in strange(ngrams(text_self[0], 1))]
            unigrams_lemma = ['-'.join(list(uni)) for uni in strange(ngrams(text_self[2], 1))]
            #tetragrams = ['-'.join(list(tetra)) for tetra in tetragrams]
            
            
            
            
            #all_grams = tetragrams + trigrams + bigrams# +  + tetragrams
            #print(all_grams[:100])
            #print(skip)
            #print(trigrams)
            #quit()unigrams_lemma, unigrams_char, trigrams, trigrams2, skip_1, skip_2, skip_3]
            input = [unigrams_lemma, unigrams_char, trigrams, trigrams2, skip_1, skip_2, skip_3]
            problem_collection[author][text_name] = input#) for tetragram in tetragrams]
            
            len_input = len(input)
            
            del input, unigrams_lemma, unigrams_char, trigrams, 
            trigrams2,skip_1, skip_2, skip_3, 
            #for gram in tetragrams:
            #    print(list(gram))
            #quit()
            #print(author,text_name,text_self[:10])
            
            del text_self, text_name
            
        del texts, author   
    #gc.collect()
      
    return problem_collection, len_input


def stats_for_ngrams_and_skipgrams(problem_collection, nb_feature_categories, frequency):
    
    flatten_all_preprocessed_text = [[] for _ in range(nb_feature_categories)]
    
    
    
    for author, texts in strange(problem_collection.items()):
        for text_name, text_self in strange(texts.items()):
            for nb_,text_ in strange(enumerate(text_self)):
                flatten_all_preprocessed_text[nb_].extend(text_)
                del text_
            del text_self, text_name
        del texts, author
       
    counters = []
    
    for c in strange(flatten_all_preprocessed_text):
        
        counters.append(Counter(c))
        del c
   
    
    trunc_words = []
    for t in strange(counters):
        tw = [k for k, _ in strange(t.most_common(frequency))]
        trunc_words.append(tw)
        del tw
        
   
    
    words_num=[]
    words_encoder = []
    for nb_, tw in strange(enumerate(trunc_words)):
        we = index_words(tw)
        #save_tools(we, os.path.join(os.getcwd(), 'words_encoder_{}'.format(nb_)))
        wn = len(we.classes_)
        words_encoder.append(we)
        words_num.append(wn)
        del we, wn, tw
    
    del counters, problem_collection, flatten_all_preprocessed_text, 
    trunc_words, nb_
    #gc.collect()
    return  words_encoder, words_num
    



def vectorise_problem_corpus(problem_collection, words_encoder, words_num, frequency, number_of_texts):
    
    
    #torch.FloatTensor(words_num[x],number_of_texts).zero_()
    freq_feature = [[] for x in range(len(words_encoder))]
    for nb1, (author, texts) in strange(enumerate(problem_collection.items())):

        for nb2, (text_name, text_self) in strange(enumerate(texts.items())):
            ngrams_ = []
          
            for nb3, text in strange(enumerate(text_self)):
             
                freq_feature[nb3].append(text)
                del text
                
            del text_name, text_self
    
        del author, texts
    
    del problem_collection, words_encoder, frequency, number_of_texts
    #gc.collect()
    
    return freq_feature, words_num


def set_sizes(problem_collection):
    
    nb_position=0
    list_of_all = []
    list_of_knows= []
    list_of_knows_names= []
    list_of_unknows= []
    for nb1, (author, texts) in strange(enumerate(problem_collection.items())):
        #break
        for nb2, (text_name, op) in strange(enumerate(texts.items())):
            
            del op
            if author != 'unknown':
                list_of_knows.append(nb_position)
                list_of_all.append(author)
                list_of_knows_names.append(author)
                #nb_position += 1
            else:
                if True:#for pair in truths_for_problem["ground_truth"]:
                    if  True:#pair['unknown-text']==text_name:
                        list_of_unknows.append(text_name)
                        list_of_all.append(text_name)
            nb_position += 1
            
            del text_name    
        del author
        
        
    l1,l2 = len(list_of_knows), len(list_of_unknows)
    
    del problem_collection, list_of_knows, list_of_unknows, list_of_all 
    #gc.collect()
    
    return l1, l2



def compute_mean_and_std(freq_feature, problem_collection, words_num):
    
    nb_position=0
    #nbunknows=0
    
    #target = [[] for x in range(len(words_num))]
    list_of_all = []
    list_of_knows= []
    list_of_knows_names= []
    list_of_unknows= []
    for nb1, (author, texts) in strange(enumerate(problem_collection.items())):
        #break
        for nb2, (text_name, op) in strange(enumerate(texts.items())):
            
            if author != 'unknown':
                list_of_knows.append(nb_position)
                list_of_all.append(author)
                list_of_knows_names.append(author)
                #nb_position += 1
            else:
                if True:#for pair in truths_for_problem["ground_truth"]:
                    if True:# pair['unknown-text']==text_name:
                        list_of_unknows.append(text_name)
                        list_of_all.append(text_name)
            nb_position += 1
            
            
            del op, text_name
        del texts, author     
    
    #print(list_of_all)
    
    #del problem_collection
    
    freq_feature_form_norm = []
    pca= []
    network_sizes = []
    for nb_, text in strange(enumerate(freq_feature)):
        
        if True:
            
            training_set = []
            test_set = []
            
            for nb2, t in strange(enumerate(text)):
                if nb2 in list_of_knows:
                    training_set.append(t)
                else:
                    test_set.append(t)
                del nb2, t
            
            del text
            
            vectorizer = TfidfVectorizer(analyzer=identity, use_idf=False, min_df=5,max_features=words_num[nb_])
            training_set = vectorizer.fit_transform(training_set).toarray()
            test_set = vectorizer.transform(test_set).toarray()
            #labels = vectorizer.vocabulary_
            scaler = StandardScaler()
            training_set = scaler.fit_transform(training_set)
            test_set = scaler.transform(test_set)
            network_sizes.append(training_set.shape[1])
            #freq_feature_form_norm.append([mean_freq, freq_std])
            #text = torch.from_numpy(text)
            #mean and std only for training !!!
            #print(labels)
            #quit()
            
            #freq = torch.cat(freq, dim=0)
            #list_of_knows = torch.LongTensor(list_of_knows)
            #list_of_unknows = torch.LongTensor(list_of_unknows)
            #freq1 = freq[list_of_knows]
            #freq2 = freq[list_of_unknows]
            #print(freq, freq1)
            #quit()
            
            #freq_ = freq.log
            #mean_freq = torch.mean(text,0)
            #freq_std = torch.std(freq, dim=0)
            #mean_freq2 = torch.mean(freq2,0)
            #freq_std2 = torch.std(freq2, dim=0)
            
            
            #print(mean_freq,freq_std)
            #quit()
            
            #freq_ = freq_ - mean_freq
            #print(freq_.norm(p=2), freq_std[freq_std==0.0])
            #freq_ = (freq_ / (1+freq_std))
            
            #print(freq_.norm(p=2))
            #U,S,V = torch.svd(freq_.t(),some=False)
            #c = torch.mm(freq_,U[:,:2])
            
            #plt.figure()
            
            #c_std = torch.std(c,dim=1)
            #variance = torch.pow(2,c_std)
            #for auto in range(c.size()[0]):
            #    plt.scatter(c[auto,0],c[auto,1],label=auto)
                #ax.annotate(auto, c[auto,0],c[auto,1]) 
            #plt.legend()
            
            #plt.savefig('pca for features')
            
            #print(c.size())#(variance/variance.sum())[:10].sum())
            #quit()
            
            #print(freq,mean_freq,freq_std)
            #quit()
            #freq_ = freq - mean_freq.expand_as(freq)
            #freq_normalised = freq_ / (freq_std.expand_as(freq))
            #freq_norm.append(freq_normalised)
            
            #nonzeroresult, _ = freq.nonzero().sort(dim=0)
            
            #f = nonzeroresult[:,1].numpy().tolist()
            
            #fc = Counter(f)
            
            #n_list = []
            #for x,y in fc.most_common():
            #    n_list.append([x,y])
                #print(x,y)
            #n_list = torch.FloatTensor(n_list)
            #_, n_idx = n_list[:,0].sort(dim=0)
            #n_list = n_list[n_idx][:,1]
            
            #idf = n_list.size()[0] / (1 + n_list.float())
            
            #if_idf = freq * idf
            #print(freq, if_idf)
            
            #quit()
            
            #print(mean_freq, freq_std)
            #quit()
            
            del vectorizer, scaler
            
            freq_feature_form_norm.append([training_set, test_set, list_of_knows_names])
            #pca.append(c)
            #pca_size.append(c.size()[1])
    #quit()
    
    del freq_feature, problem_collection, words_num#, text_name#, text_self
    #gc.collect()
    
    return freq_feature_form_norm, pca, network_sizes

def make_authors_list(problem_collection):
    
    authors_ = []
    for author, mo in strange(problem_collection.items()):
        del mo
        if author != 'unknown':
            authors_.append(author)
        del author 
    del problem_collection
    #gc.collect()
    
    return authors_

def define_model(nb_words, authors_len, freq_feature_form_norm, nb_fcat):
    
    model = StyloMeasurer(nb_words=nb_words, nb_authors=authors_len, 
                       freq_feature=freq_feature_form_norm, 
                       nb_fcat=nb_fcat#.#*len(cut_list),
                       #freq_masking = freq_masking
                       )    
    
  
    
    #for p, _ in model.named_parameters():
    #    print(p)
    del nb_words, authors_len, freq_feature_form_norm, nb_fcat
    #quit()
    #gc.collect()
    
    if use_cuda:
        model = model.cuda()
    return model    
 
 
 
        
def define_optimiser(model):
    
    optimizer = torch.optim.Adam(model.parameters(), #weight_decay = 0.001,
                               lr=0.00001)
    
    del model
    #gc.collect()
    
    return optimizer

def cluster_test(problem_collection, nb_fcat, authors, freq_feature_form_norm):
    
    
    nb_position=0
    #nbunknows=0
    
    target = [[] for x in range(nb_fcat)]
    list_of_all = []
    list_of_knows= []
    list_of_knows_names= []
    list_of_unknows= []
    for nb1, (author, texts) in enumerate(problem_collection.items()):
        #break
        for nb2, (text_name, text_self) in enumerate(texts.items()):
            
            if author != 'unknown':
                list_of_knows.append(nb_position)
                list_of_all.append(author)
                list_of_knows_names.append(author)
                #nb_position += 1
            else:
                if True:#for pair in truths_for_problem["ground_truth"]:
                    if True:# pair['unknown-text']==text_name:
                        list_of_unknows.append(text_name)
                        list_of_all.append(text_name)
            nb_position += 1
    #print(list_of_all)
    
    #quit()    
    
    
    
    #labels_for_svm = list_of_knows
    
    #print(list_of_knows)
    #print(list_of_unknows)
    #positions, names = zip(*list_of_unknows)
    #positions, names = list(positions), list(names)
    
    #print(freq_feature_form_norm)
    
    training_sets, test_sets = [], []
    feature_nb = 0
    for x in freq_feature_form_norm:
        #if feature_nb == 2:
        #    break
        feature_nb += 1
        training_set, test_set, list_of_knows_names = x
        #print(training_set.shape)
        training_sets.append(training_set)  
        test_sets.append(test_set)
        
    training_sets = numpy.concatenate(training_sets, axis=1)
    test_sets = numpy.concatenate(test_sets, axis=1)
    print(feature_nb,'gfrggdfgd')
    #all_features = torch.cat(freq_feature_form_norm, dim=1)
    
    #all_features = all_features.numpy()
    
    
    #ac = AgglomerativeClustering(n_clusters=len(authors)).fit(all_features)
    #labels_aglo = ac.labels_
    #km = AffinityPropagation().fit(unknows)
    #labels_km = km.labels_
    
    #for nb, l in enumerate(labels_aglo):
    #    print(l, list_of_all[nb])
    
    #quit()
    
    
    #knows = torch.FloatTensor(torch.zeros(len(list_of_knows), all_features.size()[1]))
    #unknows = torch.FloatTensor(torch.zeros(len(list_of_unknows), all_features.size()[1]))
    #list_of_knows = torch.LongTensor(list_of_knows)
    #list_of_unknows = torch.LongTensor(positions)
    
    
    #knows = knows.copy_(all_features[list_of_knows])
    #unknows = unknows.copy_(all_features[list_of_unknows])
    
    
    del text_name, text_self
    #knows = knows.numpy()
    #unknows = unknows.numpy()
    
    clf = OneVsRestClassifier(svm.SVC())
    clf.fit(training_set,list_of_knows_names) 
    r = clf.predict(test_set)
    answers = []
    for nb_r, r_ in enumerate(r):
        #print(r_)
        #print(list_of_unknows[nb_r])
        
        input = {}
        input["unknown-text"] = list_of_unknows[nb_r]
        input["predicted-author"] = r_
        answers.append(input)
        
    #quit()
    #gc.collect()
    
    return answers
    
    

def label_predictor(tensors_train, tensors_test, labels_train, authors_list):

    def merger(tensors):
        tensors_merged = []
        for tensor in tensors:
            #for single text sample
            tensor=torch.cat(tensor, dim=1)
            tensors_merged.append(tensor)
        return tensors_merged
    def most_common(lst):
        data = Counter(lst)
        return max(lst, key=data.get)
    
      
    random.seed(time.time())
    tensors_train = list(zip(*tensors_train))
    tensors_test = list(zip(*tensors_test))
    #random_tensor_nb = random.choice(range(len(tensors_train)))
    
    results = []
    for nb, tensor_train in enumerate(tensors_train):
        
        training = tensor_train#.numpy()
        test = tensors_test[nb]#.numpy()
    
        training = torch.cat(training, dim=0).data.cpu().numpy()
        test = torch.cat(test, dim=0).data.cpu().numpy()
    
    #print(training.size(), test.size())
    
    #quit()
    
    
    #training = torch.cat(training, dim=0)#.numpy()
    #test = torch.cat(test, dim=0)#.numpy()
    #print(training.size(),test.size(), 'blabala')
    #quit()
    
  
    
    
    
        scaler = StandardScaler()
        training_set = scaler.fit_transform(training)
        test_set = scaler.transform(test)
    
        clf = OneVsRestClassifier(svm.SVC())
        clf.fit(training_set,labels_train) 
        r = clf.predict(test_set).tolist()
        results.append(r)
        
    
    results_per_author = list(zip(*results))
    
    best_authors = []
    for author in results_per_author:
        #c = Counter(author)
        best = most_common(author)
        best_authors.append(best)
    
   
    #print(best_authors)
    
    #quit()
    
    #gc.collect()
    
    return best_authors
    

def training(m_, training_set_size, problem_collection,authors_list, loss_function1, 
             loss_function2, optimiser, freq_feature, noisy_labels):
    
    #global model
    model = m_
    model = model[1]
    #global optimiser_train, optimiser_test
    optimiser_training, optimiser_test = optimiser
    
    
    
    #counted_noisy = Counter(noisy_labels)
    
    #print(noisy_labels)
    
    #quit()
    
    
    loss_general = []
    
    acc_general_encoder = []
    acc_corrupted = []
    
    nb_authors = len(authors_list)
    
    r = None
        
 
    best_loss = None
    label_svm_pred = None
    u = -1
    alphas = []
    while True:#for _ in range(10):
        
        #alpha = sigmoid_annealing_schedule(u, 10)
        #alphas.append(alpha)
        u += 1
       
        #print(u)
      
        #alpha = sigmoid_annealing_schedule(iteritemsitemsitems_comp, 200)
        #print("alpha: {}".format(alpha))
        #alphas.append(alpha)
        #model[0].train()
        model.train()
        
        loss_comparator = 0.0
       
        acc_encoder = 0.0
        acc_corr = 0.0
        
        
        inputs = []
        corrupted = []
        list_of_names = []
        for (feat, test, names) in freq_feature:
                #print(feat)
                #quit()
            feat = torch.from_numpy(feat)
            test = torch.from_numpy(test)
            inputs.append(feat)
            corrupted.append(test)
            #del feat, test
            #list_of_names.append(names)
            #input_ = torch.zeros(feat[nb_text].size()).float()
            #input_ = input_.copy_(feat[nb_text].float())
            #inputs.append(input_)
        #inputs = torch.cat(inputs, dim=1)
        #corrupted = torch.cat(corrupted, dim=1)
        #merged = list(zip(*inputs, corrupted))
        #print(merged[0])
        #quit()
        random_x = [x for x in range(feat.size()[0])]
        #print(random_x)
        random.shuffle(random_x)
        #print(random_x)
        #quit()
        nb_ = 0
        nb2= 0
        nb3=0
        nb_corrupted = 0
        seen_test_list = []
        #authors_all = [author for author, _, _ in strange(new_generator_texts(problem_collection))]
        list_compressed_data_train = []
        list_compressed_data_test = []
        
        for x in range(feat.size()[0]):  #author, text_self, nb_text in new_generator_texts(problem_collection):
        #for author, texts in problem_collection.items():
            #print(x,nb2)
            nb2 = random_x[x]
            #x == 'unknown'
            #if x == 'unknown':
            #    unknown = True
                #continue
            #elif x != 'unknown':
            unknown = False
            
            i_s = []
            if not(unknown):
                #print('known')
                #x = names.index(author)
                for i in inputs:
                    input_ = torch.zeros(i[nb2].size()).float()
                    input_ = input_.copy_(i[nb2].float())
                    #print(input_.size())
                    i_s.append(input_)
            #
            
            else:
                pass
                #x = noisy_labels(nb_corrupted)
                #for i in corrupted:
                    #print(nb_corrupted)
                #    input_ = torch.zeros(i[nb3].size()).float()
                #    input_ = input_.copy_(i[nb3].float())
                    #print(input_.size())
                #    i_s.append(input_)
                
            #print(i_s)
            #quit()
            #if author == 'unknown':
            #    unknown = True
            #else:
            #    unknown = False
            
            #skip = numpy.random.binomial(1,0.75)
            #if skip:
                #print(skip)
            #    continue
            
            #print(len(inputs))    
            #input = freq_feature[nb_text]        
            
            
            
            #print(input)

            #quit()            
            #model[0].zero_grad()
            model.zero_grad()
            #model[0].train()
            model.train()
            #model.zero_grad()
            nb_ += 1
            #print('pair number: {}'.format(nb_))
           
            loss = 0.0
            
            #if not(unknown):
            
            #output_known, alter_label, _, auto_r1, compressed1 = training_model(i_s, unknown)#image_pairs=compressed_frags)
           
           
           
            output_unknown, alter_label, estimator, auto_r, compressed2 = model(i_s, unknown)
            
            #if not(unknown):
            #    list_compressed_data_train.append(compressed1)
            #else:
            #    list_compressed_data_test.append(compressed1)
            
            #print(output.volatile,'sfdsfgfgfghrtztrzrzrz')
            #print(len(names), nb2, len(authors_list))
            
            #for b in range(len(i_s)):
            #    
            #    target = Variable(i_s[b])
            #    if use_cuda:
            #        target = target.cuda()
                
                #print(auto_r[b].size(), target.size())    
                
            #    loss += loss_function2(auto_r[b].view(-1),target.view(-1))
                
            #loss_auto.backward()
            #optimiser_training.step()
            
            
            
            if not(unknown):
                
                
                
                author_index_known = authors_list.index(names[nb2])
                #print(author_index_known)
                one_hot = torch.FloatTensor(len(authors_list)).zero_().unsqueeze(0)
                items_id = torch.LongTensor([author_index_known]).unsqueeze(0)
                value = torch.FloatTensor([1.0]).unsqueeze(0)
                #print(freqs, items_id, one_hot)
                #freq_of_a_text = freq_feature[nb3][:,nb2].unsqueeze(0)
                #print(freq_of_a_text.size())
                authorship_vector = one_hot.scatter_(1,items_id,value).squeeze()
                
                
                
                
                
                
                author_index = Variable(torch.LongTensor([author_index_known]))
                if use_cuda:
                    author_index = author_index.cuda()
        
                loss += loss_function1(output_unknown.view(1,-1),author_index.view(-1))
        
                #loss += loss_function1(output_known.view(-1),author_index.view(-1))
            
            
            
            
            
            
            if False:
                pass# unknown:
                
               
            
            
            
            
            
            #author_index = Variable(torch.LongTensor([author_index_]))
            #if use_cuda:
            #    author_index = author_index.cuda()
            
            #loss += loss_function1(output2.view(1,-1),author_index.view(-1))/2.0
          
            
            #author_index = Variable(torch.LongTensor([guessed_label]))
            #if use_cuda:
            #    author_index = author_index.cuda()
            
            #loss += loss_function1(output3.view(1,-1),author_index.view(-1))
                
                
                
                
                
            
                
            
            
            #print(loss_comp) 
            
            #quit()
            
            
            #print(loss)
            
            
            #loss_combined = loss_comp #+ (loss_encoder_per_pair*alpha)
            if not(unknown):
                loss.backward()
            #optimizer_encoder.step()
            #if not(unknown):
            
            #else:
            optimiser_test.step()
            #optimiser_training.step()
           
           
            ##############
            
           
            teach = numpy.random.binomial(1, 0.2)
            if False:
                pass# teach and u<=3:    
               
            if not(unknown):
                #nb2 += 1
                #nb2_list.append(nb2)
                result = output_unknown.data.cpu().squeeze().topk(1,dim=0)[1][0]#.numpy().tolist()
                #print(result, out)
                #quit()
                #print(result_, author_index_)
                if result == author_index_known:
                    #print(author_index_)
                    acc_encoder += 1.0#/len(freq_feature)
                #acc_encoder += acc_encoder#/len(freq_feature)
            
         
            if not(unknown):
                loss_comparator += (loss.data.cpu().item())
            
        
          
            #for p in model.parameters():
            #    if p.grad is not None:
            #        p.grad.data.clamp_(-1.0,1.0)
            
                    
            '''optimising model'''
                            
            #optimizer_encoder.step()
            #optimiser.zero_grad()
            model.zero_grad()
            del loss, alter_label, estimator, auto_r, compressed2, output_unknown, author_index
            del i_s, author_index_known, one_hot, items_id, value, authorship_vector
                
        
     
        
        
    
        
        
        #results.append((true_positive+true_negative)/(true_positive+true_negative+false_negative+false_positive))
        #loss_general_encoder.append(loss_encoder/num_pair)
        #print(nb2, nb3)
        #quit()
        acc_general_encoder.append(acc_encoder/(x+1))
        #acc_corrupted.append(acc_corr/nb3)
        #print(nb3)
        loss_general.append(loss_comparator/nb_)  
        
        
        
        
        base1 = round((max(loss_general)-min(loss_general))/10,5)
        #base2 = round(max(results)/10,5)
        #base3 = round(max(acc_corrupted)/10,5)
        base4 = round(max(acc_general_encoder)/10,5)
    
        if base1 <= 0.0:
            base1 = max(loss_general)/10.0
        #if base2 <= 0.0:
        #    base2 = 1.0
        #if base3 <= 0.0:
        #    base3 = 1.0
        if base4 <= 0.0:
            base4 = 1.0
        #print(loss_general)
        #save_plot(loss_general, os.getcwd(), 'loss_comp_{}'.format(1), base1)
        #save_plot(results, os.getcwd(), 'dokladnosc_{}'.format(1), base2)
        #save_plot(acc_corrupted, os.getcwd(), 'accuracy_corr_{}'.format(1), base3)
        #save_plot(acc_general_encoder, os.getcwd(), 'accuracy_encoder_{}'.format(1), base4)   
        #save_plot(alphas, os.getcwd(), 'alphas_{}'.format(1), 0.1)
        #save_plot4(outputs, os.getcwd(), 'results_train_{}'.format(1))    
        
        if best_loss is None:
            best_loss = min(loss_general)
        #if best_loss_encoder is None:
            #best_loss_encoder = min(loss_general_encoder)
            
        if best_loss > min(loss_general):# or best_loss_encoder > min(loss_general_encoder):
            #torch.save(model.state_dict(), os.path.join(os.getcwd(), "model_saved_proc_{}".format(1)))
            best_loss = min(loss_general)
            #print("model saved after loss improvement")
        #if best_loss_encoder > min(loss_general_encoder):
            #torch.save(model_encoder.state_dict(), os.path.join(os.getcwd(), "model_encoder_saved_proc_{}".format(1)))
            #best_loss_encoder = min(loss_general_encoder)
            #print("encoder and comparator models saved after loss improvement")
        
        #label_svm_pred = label_predictor(list_compressed_data_train, list_compressed_data_test, names, authors_list)
        
        #if u % 1 == 0:
        #r = testing(problem_collection, model, authors_list, freq_feature, label_svm_pred)
        
        
        
        
        
        
        
        
        
    #quit()    
       
        if sum(loss_general[-1:])/1 <= 0.01*loss_general[0]:
        #if len(acc_general_encoder) == 1:
            del acc_general_encoder,loss_general,  inputs, corrupted 
            del list_of_names, freq_feature, optimiser_training, m_ 
            del optimiser_test, optimiser, problem_collection
            del feat, test, loss_function1, loss_function2
            #gc.collect()
            return model
    #return model
       
       
        
def testing(p_, m_, authors_list, f_, label_svm_pred):
    
    #global problem_collection, model
    problem_collection, model = p_, m_
    freq_feature = f_
    
    
  
    results = []
    #model[0] = model[0].eval()
    model = model.eval()
    inputs = []
    feats = []
    for (feat, test, names) in freq_feature:
                #print(feat)
                #quit()
        feat = torch.from_numpy(feat)
        test = torch.from_numpy(test)
        inputs.append(test)
        feats.append(feat)
        del feat, test
        
    
    #print(test.size()[0])
    
    #quit()
    
    svm_result = 0.0
    
    #text_names = [text_name for _, (text_name, _) in enumerate(problem_collection['unknown'].items())]
    #print(text_names)
    x = -1
    answers = []
    for text_name, _ in strange(problem_collection['unknown'].items()): #range(test.size()[0]): #x, (text_name, text_to_test) in enumerate(problem_collection['unknown'].items()):   
        x += 1
        #del mi
        #outputs[text_name] = 
        #if number_tests == 3:
        #    break
      
      
        
        #print(x)
        
        i_s = []
        for i in inputs:
            input_ = torch.zeros(i[x].size()).float()
            input_ = input_.copy_(i[x].float())
            #print(input_.size())
            i_s.append(input_)
            del input_
       
        
        _candidate_results = []
        _candidate_names = []
        
        
        
    
        
        output1, t, z, e, q = model(i_s, False)
        
        result1 = output1.data.cpu().squeeze().topk(1,dim=0)[1][0]#.numpy().tolist()
        #print(result1)
        #result2 = output2.data.cpu().squeeze().topk(1,dim=0)[1][0]
        #result3 = output3.data.cpu().squeeze().topk(1,dim=0)[1][0]
        #all_results = torch.cat(output,dim=0).sum(dim=0).max(0)[1].data[0]
        
        top_values = output1.data.cpu().squeeze().topk(3,dim=0)[0].numpy().tolist()
        #print(top_values)
        #results.append(output1)
  
        best1 = authors_list[result1]
        #best2 = authors_list[result1[1]]
        #best3 = authors_list[result1[2]]
        del i_s, output1, t, z, e, q, result1
        
        #print(best1)
        #answers = []
        #for nb1, (author, texts) in enumerate(problem_collection.items()):
        #break
        #    for nb2, (text_name, text_self) in enumerate(texts.items()):
        #        if author == 'unknown':
        
        
        input = {}
        input["unknown-text"] = text_name
        input["predicted-author"] = best1
        answers.append(input)
        #     "unknown-text": "unknown00024.txt",
        #"predicted-author": "candidate00010"
    #quit()        
    del freq_feature, inputs, feats, 
    results, model, problem_collection, i, p_, m_, f_
    #, text_name, text_to_test
    
    #gc.collect()
    
    return  answers 
        #results_general.append([text_name, _candidate_names, _candidate_results, candidate_results_singled,candidate_names_singled])   
    
        
        
        
    

