from textblob import TextBlob
import nltk
from nltk import word_tokenize, pos_tag

blob = TextBlob("Such review of risk categorisation of customers should be carried out at a periodicity of not less than once in six months.")
str = "Such review of risk categorisation of customers should be carried out at a periodicity of not less than once in six months."

nouns = []

for word,pos in blob.tags:
	if(pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
		nouns.append(word)

#print("Noun list")
#print(nouns)
#print("Noun phrases")
#print(blob.noun_phrases)

print("\n\n\nChunking output begins")


#str = "the little yellow dog barked at the cat"
str_token = word_tokenize(str)
sentence = nltk.pos_tag(str_token)

print("After pos tagging")
print(sentence)

pattern = "NP:{<JJ>*<NN>*<NN>*<NNS>*}"

NPChunker = nltk.RegexpParser(pattern)

result = NPChunker.parse(sentence)

#print(result.draw())
print(result)
print(type(result))

print("\n\n")

for i in result.subtrees(filter=lambda x: x.label() == 'NP'):
	#print(i.leaves())
	list1=[]
	for word,pos in i.leaves():
		list1.append(word)
	print(list1)
print("\n\nBye Bye")

