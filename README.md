# Language Detection Model

This Language Detection Model will be implementing a language detection module using ngram knowledge. Given a string representing some natural language utterance, the program predicts whether the text is Indonesian, Malaysian or (phonetically transcribed into English) Tamil. So given the following three strings:
<pre>
Semua manusia dilahirkan bebas dan samarata dari segi kemuliaan dan hak-hak.
Semua orang dilahirkan merdeka dan mempunyai martabat dan hak-hak yang sama.
Maitap piiviyiar cakalarum cutantiramkav piakkiaar
</pre>
... an ideal program should output/predict the following labels for the strings:
<pre>
malaysian	Semua manusia dilahirkan bebas ...
indonesian	Semua orang dilahirkan merdeka ...
tamil   	Maitap piiviyiar cakalarum cutantiramkav piakkiaar ..
</pre>

To evaluate the accuracy of your predictions, run the evaluation file eval.py:
`python3 eval.py file-containing-your-results file-containing-correct-results`

## Command
`python3 build_test_LM.py -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file`

where input-file-for-building-LM is a file given to you that contains a list of strings with their labels for you to build your ngram language models, input-file-for-testing-LM is a file containing a list of strings for you to test your language models, and output-file is a file where you store your predictions.

## Command Example:
To build and test the language model, run: \
`python3 build_test_LM.py -b input.train.txt -t input.test.txt -o input.predict.txt`

which will store your predictions in input.predict.txt. To evaluate your predictions, run:\
`python3 eval.py input.predict.txt input.correct.txt`


## Notes

### Python Version

I'm using Python Version <3.8.10> for
this assignment.

### General Notes about this assignment
For build_LM(), I noticed a lot of the code will be repetitive for each language, so I created helper functions to
handle repetitive code. 
The steps to obtain the final LM is as follow:
1) read the in_file line by line and compute a list of 4-gram tuples for each language using the compute_four_gram()
helper function. 
Computing 4-gram:
a) The beginning of the string and the end of the string are not padded, because if we are collecting
4-gram by each character instead of word, padding will not be very helpful in prediction the language. 
b) Punctuations are kept as how it is because some of these languages uses ' and -, which should not be removed.
c) Upper/lower case and numbers are kept in case specific language uses uppercase or numbers as set names or terms 
that can be used to predict the language. (e.g. countries names uppercase first character in English)
2) compute 2 dictionaries for each language. First dictionary has 4-gram as the key, and count as the value.
Second dictionary also has 4-gram as the key, but has 0 for all values. The choice for creating a count map zero
is for when we have to show the 0 entries later for add 1 smoothing
3) merge all the count maps (the current language and the zero count maps of the other two languages), showing 0 entries
4) lastly, apply add 1 smoothing and calculate the probability of each 4-gram.
5) 3 different count maps are obtained and returned for the LM: indonesian_count_map, malaysian_count_map, tamil_count_map

For test_LM(), the following steps were taken to predict the language:
1) read the in_file line by line and compute a 4-grams list for each string
2) using the 4-grams list computed, we can obtain the probabilities of each 4-gram by referring to the LM count map.
For each string and for each language, get the corresponding probability of the 4-gram, then add the log 
of each probability ,because multiplying will eventually 0 out all the probabilities
3) Now we have 3 probabilities for each language, the highest one is the language prediction
4) however, if the 4-grams in the string don't exist in the LM more than or equal to 70% of the time, the language is classified as "other".
70% is an arbitrary number, it is slightly higher since numbers and punctuation are kept.
5) If probability is 1, that means none of the 4-gram existed in the LM, language is "other"
6) write the prediction to out_file 

