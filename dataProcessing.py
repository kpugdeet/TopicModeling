from nltk.tokenize import RegexpTokenizer
import csv
import regex as re
import time
import numpy as np
import gensim
import string
import pickle
import itertools
from nltk.corpus import stopwords
from nltk.stem import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import tokenize
from nltk.parse.stanford import StanfordDependencyParser
from collections import OrderedDict
from multiprocessing import Process, Manager
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numbers import Number


# region Read OMDB data
def readDataOMDB ():
    dataFileName = './data/OMDB/moviesDetailOMDB.csv'
    dataDes = []; dataTitle = []; dataTag = []; dataID = []
    with open(dataFileName, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if len(row) > 3 and row[3].count(' ') > 3:
                dataID.append(unicode(row[0], errors='ignore').lower())
                dataTitle.append(unicode(row[1], errors='ignore').lower())
                dataTag.append(unicode(row[2], errors='ignore').lower())
                dataDes.append(unicode(row[3], errors='ignore').lower())

    # RegexpTokenizer s, pattern='\w+|\$[\d\.]+|\S+'
    tok = RegexpTokenizer(r'\w+')
    for index, des in enumerate(dataDes):
        dataDes[index] = [x for x in tok.tokenize(des) if not re.search(r'(\d)', x)]
        dataDes[index] = ' '.join(dataDes[index])

    return  dataID, dataTitle, dataTag, dataDes
# endregion

# region keyphrase_extraction_stanford_parser
def generateWordPairs (trainData, validData, testData, tfidf=True, filename='default'):
    # Initialize the Stanford Dependency Parse
    dependencyParser = StanfordDependencyParser(path_to_jar='./stanford/stanford-parser.jar', path_to_models_jar='./stanford/stanford-parser-3.7.0-models.jar')

    # Load each description into a dictionary
    trainDict = dict(); validDict = dict(); testDict = dict()
    [trainDict.update({index: des}) for index, des in enumerate(trainData)]
    [validDict.update({index: des}) for index, des in enumerate(validData)]
    [testDict.update({index: des}) for index, des in enumerate(testData)]

    trainWordPairDict = dict(); validWordPairDict = dict(); testWordPairDict = dict();
    trainDictList = list(); validDictList = list(); testDictList = list();

    wordDict, weight = generateTfidf(trainData)

    if tfidf:
        trainWordPairDict, trainDictList = multi_process_parser_tfidf(dependencyParser, trainDict, 20, wordDict, weight)
        validWordPairDict, validDictList = multi_process_parser_tfidf(dependencyParser, validDict, 20, wordDict, weight)
        testWordPairDict, testDictList = multi_process_parser_tfidf(dependencyParser, testDict, 20, wordDict, weight)
    else:
        trainWordPairDict, trainDictList = multi_process_parser(dependencyParser, trainDict, 20)
        validWordPairDict, validDictList = multi_process_parser(dependencyParser, validDict, 20)
        testWordPairDict, testDictList = multi_process_parser(dependencyParser, testDict, 20)

    trainDocList = list()
    fileOutput = open(filename + '_train_Tfidf.txt' if tfidf else filename + '_train.txt', 'w')
    for index, document in trainWordPairDict.items():
        fileOutput.write("%s\n" % document)
        trainDocList.append(document)
    print "The total number of document in the train dataset is ", len(trainWordPairDict)

    validDocList = list()
    fileOutput = open(filename + '_valid_Tfidf.txt' if tfidf else filename + '_test.txt', 'w')
    for index, document in validWordPairDict.items():
        fileOutput.write("%s\n" % document)
        validDocList.append(document)
    print "The total number of document in the valid dataset is ", len(validWordPairDict)

    testDocList = list()
    fileOutput = open(filename + '_test_Tfidf.txt' if tfidf else filename + '_test.txt', 'w')
    for index, document in testWordPairDict.items():
        fileOutput.write("%s\n" % document)
        testDocList.append(document)
    print "The total number of document in the test dataset is  ", len(testWordPairDict)

    wordPairDict = list(trainDictList)
    fileOutput = open(filename + '_wordPair_Tfidf.txt' if tfidf else filename + '_wordPair.txt', 'w')
    fileOutput.write("%s\n" % wordPairDict)
    print "The total number of word pair in the dictionary is   ", len(wordPairDict)

    return trainDocList, validDocList, testDocList, wordPairDict

def generateTfidf (trainData):
    # Remove the stop words
    stop_word_set = set(stopwords.words('english'))
    texts_without_stop = [[word_with_stop for word_with_stop in document.lower().split() if word_with_stop not in stop_word_set] for document in trainData]
    # Remove the punctuation
    exclude = set(string.punctuation)
    texts_without_punctuation = [[word_with_punctuation for word_with_punctuation in text_without_stop if word_with_punctuation not in exclude] for text_without_stop in texts_without_stop]
    # Remove the number
    texts_without_digit = [[word_with_digit for word_with_digit in text_without_punctuation if not word_with_digit.isdigit()] for text_without_punctuation in texts_without_punctuation]
    # Remove word tense
    texts_without_tense = []
    for text_without_digit in texts_without_digit:
        temp = []
        for word in text_without_digit:
            temp.append(PorterStemmer().stem(word.decode('utf-8')))
        texts_without_tense.append(temp)
    # Prepare dataset for TF-IDF processing
    temp_dataset_for_tfidf = []
    for text_without_tense in texts_without_digit:
        if len(text_without_tense) != 0:
            temp_str = ''
            for i in xrange(0, len(text_without_tense) - 1):
                temp_str += text_without_tense[i]
                temp_str += ' '
            temp_str += text_without_tense[len(text_without_tense) - 1]
            temp_dataset_for_tfidf.append(temp_str)
    # TF-IDF process
    vectorizer = CountVectorizer(decode_error="replace")
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(temp_dataset_for_tfidf))
    word = vectorizer.get_feature_names()
    word_dict = dict()
    [word_dict.update({element: index}) for index, element in enumerate(word)]
    weight = tfidf.toarray()
    return word_dict, weight

def multi_process_parser_tfidf(dependency_parser, dataset, number_of_process,word_dict,weight):
    manager = Manager()
    word_pair_dictionary = manager.dict()
    dictionary = manager.list()
    segment = len(dataset)/number_of_process
    all_processes = [Process(target=single_process_parser_tfidf, args=(dependency_parser, dataset, word_pair_dictionary,dictionary,x*segment, (x+1)*segment,word_dict,weight)) for x in xrange(0,number_of_process)]
    for p in all_processes:
        p.start()
    P_last=Process(target=single_process_parser_tfidf, args=(dependency_parser, dataset, word_pair_dictionary,dictionary,number_of_process*segment, len(dataset),word_dict,weight))
    if number_of_process*segment < len(dataset):
        P_last.start()
    for p in all_processes:
        p.join()
    if number_of_process*segment < len(dataset):
        P_last.join()
    word_pair_dictionary=OrderedDict(sorted(word_pair_dictionary.items(),key=lambda t:t[0]))
    return word_pair_dictionary,dictionary

def single_process_parser_tfidf(dependency_parser,dataset,word_pair_dictionary,dictionary,start,end,word_dict,weight):
    # Create English stop words
    stopset = stopwords.words('english')
    for doc_count in xrange(start,end):
        each_document = list()
        # seperate each pargraph into sentences
        sentences = list()
        occurance = dataset[doc_count].count(' ')
        if occurance > 30:
            previous = 0
            for x in xrange(1,int(occurance/30)):
                nth_index = find_nth_overlapping(dataset[doc_count], " ", x*30)
                sentences.append(dataset[doc_count][previous:nth_index])
                previous = nth_index
            sentences.append(dataset[doc_count][previous:])
        else:
            sentences = tokenize.sent_tokenize(dataset[doc_count])
        for sentence in sentences:
            # parse each sentence
            result = dependency_parser.raw_parse(sentence)
            dep = result.next()
            output = list(dep.triples())
            # if number of result > 0
            if len(output)>0:
                for elem in output:
                    # remove the word pair which contains the stop word
                    if elem[0][0] not in stopset and elem[2][0] not in stopset:
                        # remove the word pair by TF-IDF score
                        if check_tfidf(doc_count,word_dict,weight,elem[0][0],0.01) and check_tfidf(doc_count,word_dict,weight,elem[2][0],0.01):
                            pair = list()
                            pair.append(elem[0][0])
                            pair.append(elem[2][0])
                            # add word pair into each document collection
                            each_document.append(pair)
                            dictionary.append((elem[0][0],elem[2][0]))
        word_pair_dictionary.update({doc_count:each_document})

def multi_process_parser(dependency_parser, dataset, number_of_process):
    manager = Manager()
    word_pair_dictionary = manager.dict()
    dictionary = manager.list()
    segment = len(dataset)/number_of_process
    all_processes = [Process(target=single_process_parser, args=(dependency_parser, dataset, word_pair_dictionary,dictionary,x*segment, (x+1)*segment)) for x in xrange(0,number_of_process)]
    for p in all_processes:
        p.start()
    P_last=Process(target=single_process_parser, args=(dependency_parser, dataset, word_pair_dictionary,dictionary,number_of_process*segment, len(dataset)))
    if number_of_process*segment < len(dataset):
        P_last.start()
    for p in all_processes:
        p.join()
    if number_of_process*segment < len(dataset):
        P_last.join()
    word_pair_dictionary=OrderedDict(sorted(word_pair_dictionary.items(),key=lambda t:t[0]))
    return word_pair_dictionary,dictionary

def single_process_parser(dependency_parser,dataset,word_pair_dictionary,dictionary,start,end):
    # Create English stop words
    stopset = stopwords.words('english')
    for doc_count in xrange(start,end):
        each_document = list()
        sentences = list()
        occurance = dataset[doc_count].count(' ')
        if occurance > 30:
            previous = 0
            for x in xrange(1,int(occurance/30)):
                nth_index = find_nth_overlapping(dataset[doc_count], " ", x*30)
                sentences.append(dataset[doc_count][previous:nth_index])
                previous = nth_index
            sentences.append(dataset[doc_count][previous:])
        else:
            sentences = tokenize.sent_tokenize(dataset[doc_count])
        # iterate each sentence
        for sentence in sentences:
            # parse each sentence
            result = dependency_parser.raw_parse(sentence)
            dep = result.next()
            output = list(dep.triples())
            # if number of result > 0
            if len(output)>0:
                for elem in output:
                    # remove the word pair which contains the stop word
                    if elem[0][0] not in stopset and elem[2][0] not in stopset:
                        pair = list()
                        pair.append(elem[0][0])
                        pair.append(elem[2][0])
                        # add word pair into each document collection
                        each_document.append(pair)
                        dictionary.append((elem[0][0],elem[2][0]))
        word_pair_dictionary.update({doc_count:each_document})

def check_tfidf (currentdoc,worddict,tfidfdict,currentword,score):
    try:
        index = worddict.get(currentword,None)
        if index is not None:
            if tfidfdict[currentdoc][index]>score:
                return True
            return False
    except ValueError:
        return False
    return False

def find_nth_overlapping (haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+1)
        n -= 1
    return start
# endregion

# region tfidf_process
def tfidfProcess(trainDataList, filename='default'):
    temp_dataset_for_tfidf = save_into_temporal_file(trainDataList)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    texts_after_tfidf = tfidf_process(temp_dataset_for_tfidf, trainDataList, word2vec_model)

    fileOutput = open(filename + '_train_Tfidf2.txt', 'w')
    for item in texts_after_tfidf:
        fileOutput.write("%s\n" % item)

    # list for storing all word pairs to be the dictionary. Format will be [[a,a],[b,b],[c,c],[d,d],[e,e],[f,f]]
    train_dictionary_list = []
    for document in texts_after_tfidf:
        for word_pair in document:
            train_dictionary_list.append(word_pair)
    train_dictionary_list.sort()
    train_dictionary_list_no_duplicate = list(train_dictionary_list for train_dictionary_list, _ in itertools.groupby(train_dictionary_list))

    # for item in train_dictionary_list_no_duplicate:
    fileOutput = open(filename + '_wordPair_Tfidf2.txt', 'w')
    fileOutput.write("%s\n" % train_dictionary_list_no_duplicate)

    return texts_after_tfidf, train_dictionary_list_no_duplicate

def save_into_temporal_file(train_dataset_list):
    temp_dataset_for_tfidf = []
    for document in train_dataset_list:
        temp_str = ''
        for index in xrange(0,len(document)):
            temp_combination = document[index][0] + "aaa" + document[index][1]
            temp_str += temp_combination
            temp_str += ' '
        temp_dataset_for_tfidf.append(temp_str)
    return temp_dataset_for_tfidf

def tfidf_process(temp_dataset_for_tfidf, train_dataset_list, word2vec_model):
    vectorizer = CountVectorizer(decode_error="replace")
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(temp_dataset_for_tfidf))
    word = vectorizer.get_feature_names()
    word_dict = dict()
    word_count = 0
    for element in word:
        word_dict.update({element:word_count})
        word_count += 1
    weight = tfidf.toarray()
    # remove the word with low TF-IDF score
    texts_after_tfidf = list()
    doc_count = 0
    for document in train_dataset_list:
        temp_in_tfidf = list()
        for word_pair in document:
            temp_combination = word_pair[0] + "aaa" + word_pair[1]
            if word_pair[0] in word2vec_model.vocab and word_pair[1] in word2vec_model.vocab:
                if check_tfidf(doc_count,word_dict,weight,temp_combination,0.1):
                    temp_in_tfidf.append(word_pair)
        texts_after_tfidf.append(temp_in_tfidf)
        doc_count += 1
    return texts_after_tfidf
# endregion

# region tfidf_process_test_dataset
def tfidfProcess2(trainDataList, dataList, filename='default', typeSelect='default'):
    final_test_result = list()
    count = 0
    for test_document in dataList:
        temp_dataset_for_tfidf = save_into_temporal_file2(trainDataList, test_document)
        texts_after_tfidf = tfidf_process2(temp_dataset_for_tfidf, test_document)
        final_test_result.append(texts_after_tfidf)
        count += 1
    fileOutput = open(filename + '_' + typeSelect + '_Tfidf2.txt', 'w')
    for item in final_test_result:
        fileOutput.write("%s\n" % item)
    return final_test_result

def save_into_temporal_file2(train_dataset_list, test_document):
    temp_dataset_for_tfidf = []
    # process train dataset
    for document in train_dataset_list:
        train_temp_str = ''
        for index in xrange(0,len(document)-1):
            temp_combination = document[index][0] + "aaa" + document[index][1]
            train_temp_str += temp_combination
            train_temp_str += ' '
            train_temp_str += document[len(document)-1][0] + "aaa" + document[len(document)-1][1]
        temp_dataset_for_tfidf.append(train_temp_str)
    # process test document and add it into train dataset
    test_temp_str = ''
    for index in xrange(0,len(test_document)-1):
        temp_combination = test_document[index][0] + "aaa" + test_document[index][1]
        test_temp_str += temp_combination
        test_temp_str += ' '
    test_temp_str += test_document[len(test_document)-1][0] + "aaa" + test_document[len(test_document)-1][1]
    temp_dataset_for_tfidf.append(test_temp_str)
    return temp_dataset_for_tfidf

def tfidf_process2(temp_dataset_for_tfidf, test_document):
    # TF-IDF process
    vectorizer = CountVectorizer(decode_error="replace")
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(temp_dataset_for_tfidf))
    word = vectorizer.get_feature_names()
    word_dict = dict()
    word_count = 0
    for element in word:
        word_dict.update({element:word_count})
        word_count += 1
    weight = tfidf.toarray()
    # remove the word with low TF-IDF score
    texts_after_tfidf = list()
    temp_in_tfidf = list()
    for word_pair in test_document:
        temp_combination = word_pair[0] + "aaa" + word_pair[1]
        if check_tfidf(len(temp_dataset_for_tfidf)-1,word_dict,weight,temp_combination,0.1):
            temp_in_tfidf.append(word_pair)
    texts_after_tfidf.append(temp_in_tfidf)
    return texts_after_tfidf
# endregion

# region word_pair_cluster
def wordPairCluster (trainDataList, k=1000):
    train_dataset_set = from_list_into_set(trainDataList)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    word_vector_dictionary, word_list = convert_dictionary_into_vector(word2vec_model, train_dataset_set)
    return kmean_cluster(word_vector_dictionary, word_list, k)

def from_list_into_set(trainDataList):
    temp_set = set()
    for document in trainDataList:
        for word in document.split():
            temp_set.add(word)
    return temp_set

def kmean_cluster (word_vector_dictionary, word_list, k):
    group_dictionary = dict()
    numberLoop = np.array([k])
    np_word_vector_dictionary = np.array(word_vector_dictionary)
    for n_clusters in numberLoop:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np_word_vector_dictionary)
        silhouette_avg = silhouette_score(np_word_vector_dictionary, kmeans.labels_,sample_size=50)
        print('For n_clusters = {0} The average silhouette_score is : {1}'.format(n_clusters, silhouette_avg))
    cluster_to_words = find_word_clusters(word_list, kmeans.labels_)
    for c in cluster_to_words:
        group_dictionary.update({c:list(cluster_to_words[c])})
    return group_dictionary

def convert_dictionary_into_vector (word2vec_model, original_dictionary):
    temp_vector_list = list()
    temp_word_list = list()
    for word in original_dictionary:
        if word in word2vec_model.vocab:
            temp_vec = word2vec_model[word]
            temp_vector_list.append(temp_vec)
            temp_word_list.append(word)
    return temp_vector_list,temp_word_list

def find_word_clusters (labels_array, cluster_labels):
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words

class autovivify_list (dict):
    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError
# endregion

# region generate_word_count
def wordCount (trainData, validData, testData, wordPair, group_Dict, filename='default', mode='kmean'):
    if mode != 'kmean':
        group_Dict = pickle.load(open('represented_word_corresponding_relationship.pkl', 'rb'))

    # process dictionary, convert real word into group index and remove duplicate
    group_word_dictionary = multi_process_group_word_converter(wordPair, 8, group_Dict, mode)
    print 'The length of original dict is:  ', len(wordPair)
    print 'The length of group dict is:     ', len(group_word_dictionary)
    group_word_dictionary.sort()
    group_word_dictionary_without_duplicate = list(group_word_dictionary for group_word_dictionary, _ in itertools.groupby(group_word_dictionary))
    print 'The length of group dict set is: ', len(group_word_dictionary_without_duplicate)
    with open(filename + '_groupWordDict.pkl', 'w') as f:
        pickle.dump(group_word_dictionary_without_duplicate, f)

    # Train
    train_dataset_word_count_list = multi_process_word_count(trainData, 8, group_word_dictionary_without_duplicate,group_Dict,mode)
    train_dataset_word_count_path = open(filename + '_trainWC.txt', 'w')
    for item in train_dataset_word_count_list:
        train_dataset_word_count_path.write("%s\n" % item)
    with open(filename + '_trainWC.pkl', 'w') as f:
        pickle.dump(train_dataset_word_count_list, f)

    # Valid
    valid_dataset_word_count_list = multi_process_word_count(validData, 8, group_word_dictionary_without_duplicate,group_Dict,mode)
    valid_dataset_word_count_path = open(filename + '_validWC.txt', 'w')
    for item in valid_dataset_word_count_list:
        valid_dataset_word_count_path.write("%s\n" % item)
    with open(filename + '_validWC.pkl', 'w') as f:
        pickle.dump(valid_dataset_word_count_list, f)

    # Test
    test_dataset_word_count_list = multi_process_word_count(testData, 8, group_word_dictionary_without_duplicate,group_Dict,mode)
    test_dataset_word_count_path = open(filename + '_testWC.txt', 'w')
    for item in test_dataset_word_count_list:
        test_dataset_word_count_path.write("%s\n" % item)
    with open(filename + '_testWC.pkl', 'w') as f:
        pickle.dump(test_dataset_word_count_list, f)

    print 'Number of train document:  ', len(train_dataset_word_count_list)
    print 'Number of valid document:  ', len(valid_dataset_word_count_list)
    print 'Number of test document:   ', len(test_dataset_word_count_list)
    print 'Number of words in dictionary: ', len(group_word_dictionary_without_duplicate)

def multi_process_group_word_converter(dataset,number_of_process,group_dict,mode):
    manager = Manager()
    group_word_collection = manager.list()
    segment = len(dataset)/number_of_process
    all_processes = [Process(target=single_process_group_word_converter, args=(dataset, group_word_collection,x*segment, (x+1)*segment,group_dict,mode)) for x in xrange(0,number_of_process)]
    for p in all_processes:
        p.start()
    P_last=Process(target=single_process_group_word_converter, args=(dataset, group_word_collection,number_of_process*segment, len(dataset),group_dict,mode))
    if number_of_process*segment < len(dataset):
        P_last.start()
    for p in all_processes:
        p.join()
    if number_of_process*segment < len(dataset):
        P_last.join()
    return group_word_collection

def single_process_group_word_converter(dataset,group_word_collection,start,end,group_dict,mode):
    if mode == 'kmean':
        for word_pair_index in xrange(start,end):
            # iterate each word pair in training dictionary
            group1 = ''
            group2 = ''
            # iterate each word group in represented word relationships
            for index,group_word in group_dict.items():
                # if a word in specific word group
                if dataset[word_pair_index][0] in group_word:
                    group1 = index
                if dataset[word_pair_index][1] in group_word:
                    group2 = index
            pair = list()
            pair.append(group1)
            pair.append(group2)
            group_word_collection.append(pair)
    if mode == 'wordnet':
        for word_pair_index in xrange(start,end):
            # iterate each word pair in training dictionary
            group1 = dataset[word_pair_index][0]
            group2 = dataset[word_pair_index][1]
            # if a word in specific word group
            if group_dict.has_key(dataset[word_pair_index][0]):
                group1 = group_dict[dataset[word_pair_index][0]]
            if group_dict.has_key(dataset[word_pair_index][1]):
                group2 = group_dict[dataset[word_pair_index][1]]
            pair = list()
            pair.append(group1)
            pair.append(group2)
            group_word_collection.append(pair)

def multi_process_word_count(dataset,number_of_process,word_dict,group_dict,mode):
    manager = Manager()
    word_pair_collection = manager.dict()
    segment = len(dataset)/number_of_process
    word_count_list = list()
    all_processes = [Process(target=single_process_word_count, args=(dataset, word_pair_collection,x*segment, (x+1)*segment,word_dict,group_dict,mode)) for x in xrange(0,number_of_process)]
    for p in all_processes:
        p.start()
    P_last=Process(target=single_process_word_count, args=(dataset, word_pair_collection,number_of_process*segment, len(dataset),word_dict,group_dict,mode))
    if number_of_process*segment < len(dataset):
        P_last.start()
    for p in all_processes:
        p.join()
    if number_of_process*segment < len(dataset):
        P_last.join()
    word_pair_collection_dict = word_pair_collection
    word_pair_collection_dict=OrderedDict(sorted(word_pair_collection_dict.items(),key=lambda t:t[0]))
    for key, elem in word_pair_collection_dict.items():
        word_count_list.append(elem)
    return word_count_list

def single_process_word_count(dataset,word_pair_collection,start,end,word_dict,group_dict,mode):
    if mode == "kmean":
        for doc_count in range(start,end):
            temp_dataset_list = list()
            for word_pair_in_dataset in dataset[doc_count]:
                group1 = ''
                group2 = ''
                for index,group_word in group_dict.items():
                    # if a word in specific word group
                    if word_pair_in_dataset[0] in group_word:
                        group1 = index
                    if word_pair_in_dataset[1] in group_word:
                        group2 = index
                pair = list()
                pair.append(group1)
                pair.append(group2)
                temp_dataset_list.append(pair)
            document_word_count_temp = list()
            # print temp_dataset_list
            for word_pair in word_dict:
                frequency = temp_dataset_list.count(word_pair)
                document_word_count_temp.append(frequency)
            word_pair_collection.update({doc_count:document_word_count_temp})
    if mode == "wordnet":
        for doc_count in range(start,end):
            temp_dataset_list = list()
            for word_pair_in_dataset in dataset[doc_count]:
                group1 = word_pair_in_dataset[0]
                group2 = word_pair_in_dataset[1]
                # if a word in specific word group
                if group_dict.has_key(word_pair_in_dataset[0]):
                    group1 = group_dict[word_pair_in_dataset[0]]
                if group_dict.has_key(word_pair_in_dataset[1]):
                    group2 = group_dict[word_pair_in_dataset[1]]
                pair = list()
                pair.append(group1)
                pair.append(group2)
                temp_dataset_list.append(pair)
            document_word_count_temp = list()
            # print temp_dataset_list
            for word_pair in word_dict:
                frequency = temp_dataset_list.count(word_pair)
                document_word_count_temp.append(frequency)
            word_pair_collection.update({doc_count:document_word_count_temp})
# endregion


if __name__ == '__main__' :
    start = time.time()
    # Read Data
    dataID, dataTitle, dataTag, dataDes = readDataOMDB()
    print 'Data dimension: ', len(dataID), len(dataTitle), len(dataTag), len(dataDes)

    # Generate word pair with stanford parse
    print 1
    trainWP, validWP, testWP, WPDict = generateWordPairs (dataDes[:5000], dataDes[5000:6000], dataDes[6000:], tfidf=True, filename='./input/OMDB')

    # Second Tfidf for train dataset
    print 2
    trainWP2, WPDict2 = tfidfProcess(trainWP, filename='./input/OMDB')

    # Second Tfidf for valid dataset
    print 3
    validWP2 = tfidfProcess2(trainWP2, validWP, filename='./input/OMDB', typeSelect='valid')

    # Second Tfidf for test dataset
    print 4
    testWP2 = tfidfProcess2(trainWP2, testWP, filename='./input/OMDB', typeSelect='test')

    # K-Mean Cluster
    print 5
    groupDict = wordPairCluster(dataDes[:5000], k=10)

    # Generate word count
    print 6
    wordCount(trainWP2, validWP2, testWP2, WPDict2, groupDict, filename='./input/OMDB', mode='kmean')

    # Timer
    print (time.time()-start)