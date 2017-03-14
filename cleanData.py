#For generating the japanese resturants reivews

import json
import nltk
import pickle
import re
from nltk import word_tokenize
from nltk.tag import StanfordPOSTagger
import time

#create reviews about japanese resturants
def loadData(filePath,bus_cat_dict):

    business_dict = {}
    # open the file that contains data to clean
    f = open(filePath)
    # initiate english vocab list
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    i = 0
    # open file to write, wirte the reviews to the train data txt
    f2 = open('./Data/JapaneseR_no_punc.txt','w')
    # open file to write, wirte the reviews to the txt file and use part of it as test data
    f3 = open('./Data/JapaneseR_with_punc.txt', 'w')
    # create a dictionary that store the 'NN' word
    nn_word_dict = {}
    # set a timer
    start = time.time()
    while True:
        # read the file line by line
        lineStr = f.readline()

        if lineStr:
            # load the data
            data = json.loads(lineStr)
            # announce business id
            business_id = data['business_id']

            # announce business name
            bus_name = bus_cat_dict[business_id][1]

            #announce business city
            city_name = bus_cat_dict[business_id][2]

            # announce business category
            cate_name = bus_cat_dict[business_id][0]

            # announce business review count
            reviews_count = bus_cat_dict[business_id][3]

            # select the city, do not include the citys that are from american and canada
            if 'Karlsruhe' not in city_name and 'Edinburgh' not in city_name:

                # if the key is not in the dictionary then announce a dicitoary that contains business information
                if business_id not in business_dict and ('Japanese' in cate_name and 'Restaurants' in cate_name):

                    business_dict[business_id] = {'name': bus_name,'review_num': reviews_count, 'city' : city_name, 'category': cate_name, 'reviews' : [] }

                elif ('Japanese' in cate_name and 'Restaurants' in cate_name):

                    # clean the review and make them lower case for training the model
                    review_sentence = data['text'].replace('\n',' ').replace(")",'').replace("(",'').replace("%",'').lower()



                    #add the reviews into the dictionary
                    business_dict[business_id]['reviews'].append(review_sentence)

                    # replace the restaurant name with 'RESTAURANT_NAME'
                    new_review_sentence = review_sentence.replace(bus_name, "RESTAURANT_NAME")



                    # delete short sentence
                    if len(nltk.word_tokenize(new_review_sentence)) >= 6:

                        # delete Punctuation like question marks and so on, and leave the underscore
                        del_punc_review_sentence = re.sub('[^A-Za-z_]+', ' ', new_review_sentence)

                        #check whether the sentence is english or not
                        if check_english(nltk.word_tokenize(del_punc_review_sentence), english_vocab):

                            #write every review into the txt file
                            f2.write(del_punc_review_sentence + '\n')

                            #write another that includes the punctuation
                            f3.write(new_review_sentence + '\n')
                        else:
                            print(del_punc_review_sentence)


            # a timer
            i += 1
            if i % 28000 ==0:
                print(i/28000,'%')
                end = time.time()
                elapsed = end - start
                print("Time taken : ", elapsed, "seconds.")

        else:
            break
    f.close()
    f2.close()
    f3.close()

    return business_dict, nn_word_dict

#the function is used to create a dictionary that contains the business data that used in other functions
def create_bus_dict(businessFliePath):
    bus_cat_dict = {}
    f2 = open(businessFliePath)
    while True:
        lineStr = f2.readline()

        if lineStr and lineStr and lineStr != '':
            bus_data = json.loads(lineStr)

            bus_cat_dict[bus_data['business_id']] = [bus_data['categories'], bus_data['name'].lower(), bus_data['city'], bus_data['review_count']]

        else:
            break
    f2.close()
    print('finish bus-cat')
    return bus_cat_dict

def check_english(input_list,english_vocab):

    sentence_len = len(input_list)
    if sentence_len <= 1:
        return False


    i = 0.0

    for words in input_list:
        if words in english_vocab:
            i = i + 1

    if i/float(sentence_len) > 0.4:
        return True
    else:
        return False



#delete the bussiness that are not related the food
def delete_unrelated_bus(businessFliePath,business_dict):
    f2 = open(businessFliePath)
    c_list =[]
    while True:
        lineStr = f2.readline()
        if lineStr and lineStr != '':
            bus_data = json.loads(lineStr)
            if 'Food' not in bus_data['categories'] and 'Restaurants' not in bus_data['categories']:
                #print(data['categories'])
                #c_list += bus_data['categories']
                if bus_data['business_id'] in business_dict:
                    del business_dict[bus_data['business_id']]

            elif bus_data['business_id'] in business_dict: #add company name to the dictionary
                business_dict[bus_data['business_id']]['name'] = bus_data['name']
                business_dict[bus_data['business_id']]['review_num'] = len(business_dict[bus_data['business_id']]['reviews'])
        else:
            break
    return business_dict

#The function will return an adjust sentences if there are two or more 'NN' words connected or the original sentences
#The function will return a dictionay that key is 'word', and the value is 'NN' or 'NNS'



if __name__ == '__main__':

    # create a related business dictionary, a related word dictionary
    clean_bus_dict, nn_word_dict = loadData('Data/yelp_academic_dataset_review.json',
                                            create_bus_dict('./Data/yelp_academic_dataset_business.json'))
    num = 0
    for element in clean_bus_dict:
        print(clean_bus_dict[element]['review_num'])
        num = num + int(clean_bus_dict[element]['review_num'])

    print(num)
    print(nn_word_dict)


