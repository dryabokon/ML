# ----------------------------------------------------------------------------------------------------------------------
import nltk
import re
import spacy
from spacy.lang.en import English
#from spacy.kb import KnowledgeBase
from spacy import displacy
import warnings
# ----------------------------------------------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
# ----------------------------------------------------------------------------------------------------------------------
text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on, Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""
# ----------------------------------------------------------------------------------------------------------------------
def tokenize_split(text):
    tokens = text.split()
    return tokens
# ----------------------------------------------------------------------------------------------------------------------
def tokenize_re(text):
    tokens = re.findall("[\w']+", text)
    return tokens
# ----------------------------------------------------------------------------------------------------------------------
def tokenize_nltk(text):
    tokens = nltk.word_tokenize(text.lower())
    return tokens
# ----------------------------------------------------------------------------------------------------------------------
def tokenize_spacy(text):
    nlp = English()
    tokens = []
    for token in nlp(text):
        tokens.append(token.text)
    return tokens
# ----------------------------------------------------------------------------------------------------------------------
def sentence_tokenize_re(text):
    sentences = re.compile('[.!?] ').split(text)
    return sentences
# ----------------------------------------------------------------------------------------------------------------------
def sentence_tokenize_nltk(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    return sentences
# ----------------------------------------------------------------------------------------------------------------------
def sentence_tokenize_spacy(text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    sentences=[]
    for sent in nlp(text).sents:
        sentences.append(sent.text)

    return sentences
# ----------------------------------------------------------------------------------------------------------------------
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text
# ----------------------------------------------------------------------------------------------------------------------
def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    result  = [token for token in tokens if token not in stopword_list]
    return result
# ----------------------------------------------------------------------------------------------------------------------
def do_stemming(tokens):
    ps = nltk.porter.PorterStemmer()
    new_tokens = [ps.stem(word) for word in tokens]
    return new_tokens
# ----------------------------------------------------------------------------------------------------------------------
def do_lemmatize(text):
    nlp = English()
    tokens = nlp(text)
    new_tokens = [word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in tokens]
    return new_tokens
# ----------------------------------------------------------------------------------------------------------------------
def get_entities(text):

    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(text)
    entities = [token.ent_type_ for token in tokens]
    return entities
# ----------------------------------------------------------------------------------------------------------------------
def example_similarity(text):

    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(text)

    for i in range(len(tokens)-1):
        for j in range(i+1,len(tokens)):
            print(tokens[i].text, tokens[j].text, tokens[i].similarity(tokens[j]))
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_displacy_dependency(text):
    #'http://127.0.0.1:5000/'
    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(text)
    displacy.serve(tokens, style="dep")


    return
# ----------------------------------------------------------------------------------------------------------------------
def example_displacy_entities(text):
    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(text)
    displacy.serve(tokens, style="ent")
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    nlp = spacy.load('en_core_web_sm')
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=3)

    # adding entities
    kb.add_entity(entity="Q1004791", freq=6, entity_vector=[0, 3, 5])
    kb.add_entity(entity="Q42", freq=342, entity_vector=[1, 9, -3])
    kb.add_entity(entity="Q5301561", freq=12, entity_vector=[-2, 4, 2])

    # adding aliases
    kb.add_alias(alias="Douglas", entities=["Q1004791", "Q42", "Q5301561"], probabilities=[0.6, 0.1, 0.2])

    candidates = kb.get_candidates("Douglas")
    for c in candidates:
        print(" ", c.entity_, c.prior_prob, c.entity_vector)
