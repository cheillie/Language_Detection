#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import nltk
import sys
import getopt
import math

def compute_four_gram(stripped_line, four_gram_list):
    """
    compute a list of four grams in four_gram_list given stripped_line
    The beginning of the string and the end of the string are not padded,
    punctuations, upper/lower case and numbers are kept as it is
    """
    # compute a list of 4-grams where the units are strings (e.g. ['Mesk', 'ipun', ' beg'])
    outputs = []
    for i in range(len(stripped_line) - 4):
        outputs.append(stripped_line[i:i+4])
    
    # compute a list of 4-grams tuples where the units are characters
    # (e.g. [('M', 'e', 's', 'k'), ('e', 's', 'k', 'i')])
    for output in outputs:
        four_gram = tuple(output)
        four_gram_list.append(four_gram)

def compute_count_map(four_gram):
    """
    given a 4-grams list, compute a count map dictionary where the key is the 4-gram,
    and value is the count of the 4-gram
    """
    count_map = {}
    for i in four_gram:
        count_map[i] = count_map.get(i, 0) + 1
    return count_map

def compute_count_map_zero(four_gram):
    """
    given a 4-grams list, compute a zero count map dictionary where the key is the 4-gram,
    and value is all 0. This zero count map will be used to display zero entries
    """
    count_map_zero = {}
    for i in four_gram:
        count_map_zero[i] = count_map_zero.get(i, 0)
    return count_map_zero
    
def calculate_probability(count_map):
    """
    apply add 1 smoothing and calculate the probability of each 4-gram
    """
    # apply add 1 smoothing
    for gram in count_map: 
        count_map[gram] += 1
    # compute the total 4-gram count
    total_count = sum(count_map.values())

    # calculate probability of each 4-gram by dividing
    # the count of the 4-gram by the total count of all 4-grams
    for gram in count_map: 
        count_map[gram] = count_map[gram]/total_count

def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
    print("building language models...")

    indonesian_four_gram = []
    malaysian_four_gram = []
    tamil_four_gram = []
    
    # read the in_file line by line
    file = open(in_file, 'r')
    lines = file.readlines()

    for line in lines:
        label = line.split()[0]
        if label == "indonesian":
            # strip the label off each line
            stripped_line = line.strip("indonesian ")
            # compute 4-grams list
            compute_four_gram(stripped_line, indonesian_four_gram)
        elif label == "malaysian":
            stripped_line = line.strip("malaysian ")
            compute_four_gram(stripped_line, malaysian_four_gram)
        else:
            stripped_line = line.strip("tamil ")
            compute_four_gram(stripped_line, tamil_four_gram)

    # compute a count map dictoinary for each language where the key is the 4-gram, and the value is the counts
    # count_map_zero has all value count at 0, used for add 1 smoothing later
    indonesian_count_map = compute_count_map(indonesian_four_gram)
    indonesian_count_map_zero = compute_count_map_zero(indonesian_four_gram)

    malaysian_count_map = compute_count_map(malaysian_four_gram)
    malaysian_count_map_zero = compute_count_map_zero(malaysian_four_gram)

    tamil_count_map = compute_count_map(tamil_four_gram)
    tamil_count_map_zero = compute_count_map_zero(tamil_four_gram)

    # merge count map with the other two zero count map, so that zero entries are shown in each LM
    indonesian_count_map = {**indonesian_count_map, **tamil_count_map_zero, **malaysian_count_map_zero}
    malaysian_count_map = {**malaysian_count_map, **tamil_count_map_zero, **indonesian_count_map_zero}
    tamil_count_map = {**tamil_count_map, **indonesian_count_map_zero, **malaysian_count_map_zero}

    # modify count map dictionary values with the probability of each 4-gram
    calculate_probability(indonesian_count_map)
    calculate_probability(malaysian_count_map)
    calculate_probability(tamil_count_map)

    return indonesian_count_map, malaysian_count_map, tamil_count_map

def test_LM(in_file, out_file, LM):
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print("testing language models...")

    indonesian_LM = LM[0]
    malaysian_LM = LM[1]
    tamil_LM = LM[2]

    # clear the out_file
    open(out_file, 'w').close()

    # read the in_file line by line
    file = open(in_file, 'r')
    lines = file.readlines()

    for line in lines:
        # initialize probability to 1
        indonesian_prob = 1
        malaysian_prob = 1
        tamil_prob = 1

        # compute the 4-grams list for each string
        four_grams = []
        compute_four_gram(line, four_grams)

        num_gram = len(four_grams)
        none_count = 0
        for four_gram in four_grams:
            # obtain the probability of the 4-grams in the current string for each language
            indonesian_value = indonesian_LM.get(four_gram)
            malaysian_value = malaysian_LM.get(four_gram)
            tamil_value = tamil_LM.get(four_gram)

            # count the times the 4-gram doesn't exist in any LM (for classifying 'other')
            # only need to use one of the three languages because as long as one is None,
            # that means it never showed up in any languages
            if indonesian_value == None:
                none_count+=1

            # if the 4-gram value exists, sum the log of each probability in the string to predict the language
            if indonesian_value != None:
                indonesian_prob += math.log(indonesian_value)
            if malaysian_value != None:
                malaysian_prob += math.log(malaysian_value)
            if tamil_value != None:
                tamil_prob += math.log(tamil_value)
        highest_prob = max([indonesian_prob, malaysian_prob, tamil_prob])

        # write prediction to out_file
        f = open(out_file, "a")

        # if n-gram doesn't exist in LM more than or equal to 70% of the time, language is other
        if (none_count/num_gram >= 0.7):
            f.write("other " + line)
        # if probability is 1, that means none of the 4-gram existed in the LM, language is other
        elif highest_prob == 1:  
            f.write("other " + line)
        elif highest_prob == indonesian_prob:
            f.write("indonesian " + line)
        elif highest_prob == malaysian_prob:
            f.write("malaysian " + line)
        elif highest_prob == tamil_prob:
            f.write("tamil " + line)
        f.close()


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file"
    )


input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], "b:t:o:")
except getopt.GetoptError:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == "-b":
        input_file_b = a
    elif o == "-t":
        input_file_t = a
    elif o == "-o":
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
test_LM(input_file_t, output_file, LM)
