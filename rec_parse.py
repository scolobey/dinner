import csv
import numpy as np

phrase_key = {}
phrase_key_index = 0

def parse_recipes():
    global phrase_key
    global phrase_key_index

    with open("rec_magic.csv", 'r') as f:
        data = list(csv.reader(f, delimiter=","))
        recipes = []
        data = data[1:]
        for x in data:
            phrase_set = []
            for y in x[3:]:
                phrase_set = phrase_set + (y[1:-1].split(","))
            parsed_set = []
            for z in phrase_set:
                if z in phrase_key:
                    parsed_set.append(phrase_key[z])
                else:
                    parsed_set.append(phrase_key_index)
                    phrase_key[z] = phrase_key_index
                    phrase_key_index = phrase_key_index + 1
            recipes.append(parsed_set)

    trainX = np.zeros((len(data), phrase_key_index), dtype=np.int)

    for i, rec in enumerate(recipes):
        for j, phrase in enumerate(rec):
            trainX[i][phrase] += 1

    return trainX

def parse_newest_recipes():
    global phrase_key
    global phrase_key_index

    with open("rec_recent_1000.csv", 'r') as f:
        data = list(csv.reader(f, delimiter=","))
        data = data[1:] 
        recipes = []

        for x in data:
            phrase_set = []
            clipped = x[3:]
            parsed_set = []
            
            for y in clipped:
                phrase_set = phrase_set + (y[1:-1].split(","))
            
            for z in phrase_set:
                if z in phrase_key:
                    parsed_set.append(phrase_key[z])
            recipes.append(parsed_set)

    trainX = np.zeros((len(data), phrase_key_index), dtype=np.int)

    for i, rec in enumerate(recipes):
        for j, phrase in enumerate(rec):
            trainX[i][phrase] += 1

    return trainX

def convert_to_id(id_set):
    recipe_set = []
    with open('rec_recent_1000.csv') as csvfile:
        
        reader = list(csv.reader(csvfile, delimiter=","))
        reader = reader[1:]
        for i, row in enumerate(reader):
            if i in id_set:
                recipe_set.append(row)
    return recipe_set

# accept an array containing arrays of integers with equal length and reserve every 10th entry as test data.
def split(list):
    dataLength = len(list[0])
    testLength = len(list)//10+1
    trainLength = len(list)-testLength

    trainX = np.zeros((trainLength, dataLength), dtype=np.int)
    testX = np.empty((testLength, dataLength), dtype=np.int)

    artificial_train_index = 0
    artificial_test_index = 0
    for i, recipe in enumerate(list):
        if i%10 == 0:
            testX[artificial_test_index] = recipe
            artificial_test_index += 1
        else:
            trainX[artificial_train_index] = recipe
            artificial_train_index += 1
    return trainX, testX
