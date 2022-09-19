import numpy
import os
import random

from numpy.random import default_rng

def read_words_and_triggers(additional_path='', return_questions=False):

    words_path = os.path.join(additional_path, 'chosen_features.txt')
    with open(words_path, encoding='utf8') as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]

    # Checking all's fine with the stimuli
    assert len(lines) == 32
    animals = {l[0] : [f.replace('_', ' ') for f in l[1:]][:3] for l in lines[:16]}
    objects = {l[0] : [f.replace('_', ' ') for f in l[1:]][:3] for l in lines[16:]}

    assert len(animals) == len(objects)
    animal_feat_num = max([len(v) for k, v in animals.items()])
    object_feat_num = max([len(v) for k, v in objects.items()])
    assert animal_feat_num == object_feat_num

    # Finalizing the list of stimuli
    animals_and_objects = animals.copy()
    animals_and_objects.update(objects)

    # Generating triggers

    word_to_trigger = {w : w_i+1 for w_i, w in enumerate(animals_and_objects.keys())}
    word_to_trigger.update({'' : len(word_to_trigger)+1})
    #print(word_to_trigger)

    if return_questions:
        return word_to_trigger, animals_and_objects
    else:
        return word_to_trigger

def generate_runs(animals_and_objects):
    
    words = list(animals_and_objects.keys())
    assert len(words) == 32
    
    right_or_wrong = [False for i in range(16)] + [True for i in range(16)]
    rng = default_rng()
    
    runs_words = list()
    runs_questions = list()
    runs_answers = list()
    
    for i in range(24):
        
        empty_stimulus = random.randint(0, 32)
        
        current_answers = list(random.sample(list(right_or_wrong), k=32))
        current_answers.insert(empty_stimulus, False)
        
        current_word_indices = list(random.sample(list(range(32)), k=32))
        current_words = [words[i] for i in current_word_indices]
        
        current_word_indices.insert(empty_stimulus, random.randint(0, 32))
        current_words.insert(empty_stimulus, '')
        
        current_words = numpy.array(current_words)
        current_answers = numpy.array(current_answers)
        assert current_words.shape == (33, )
        assert current_answers.shape == (33, )
        runs_words.append(current_words)
        runs_answers.append(current_answers)
        
        current_questions = list()
        
        for q_i, q in enumerate(current_answers):
            
            # Selecting the question out of the two possibilities
            selection = random.randint(0, 1)
            
            # question whose answer is True
            if q == True:
                word = current_words[q_i]
            
            # question whose answer is False
            else:
                word_index = current_word_indices[q_i]
                # Sampling the question from the same broad category (animals vs objects)
                possible_words = list(range(16)) if word_index <= 15 else list(range(16, 32))
                possible_words = [words[i] for i in possible_words if i != word_index]
                word = random.sample(possible_words, k=1)[0]
                #print(word)
            
            q_selec = animals_and_objects[word][selection]
            current_questions.append(q_selec)
        
        current_questions = numpy.array(current_questions)
        assert current_questions.shape == (33, )
        runs_questions.append(current_questions)
    
    runs_words = numpy.array(runs_words)
    runs_questions = numpy.array(runs_questions)
    runs_answers = numpy.array(runs_answers)
    for data in [runs_words, runs_questions, runs_answers]:
        assert data.shape == (24, 33)
    
    return runs_words, runs_questions, runs_answers
