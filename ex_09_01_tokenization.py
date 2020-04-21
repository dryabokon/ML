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
the_text = "At Unicsoft, we embody learning and caring culture, attracting the best talent, who stay with us and our customers for years. That's how we live, learn, grow, and succeed together. We are motivated experts seeking out challenges. We look for partners, rather than just customers."
#the_text = "Gerard (died 1108) was Archbishop of York between 1100 and 1108 and Lord Chancellor of England from 1085 until 1092. A Norman, he was a member of the cathedral clergy at Rouen before becoming a royal clerk under King William I of England, who appointed him Lord Chancellor. He continued in that office under King William II Rufus, who rewarded him with the Bishopric of Hereford in 1096. Soon after Henry I's coronation, Gerard was appointed to the recently vacant see of York, and became embroiled in the dispute between York and the see of Canterbury concerning which archbishopric had primacy over England. He secured papal recognition of York's jurisdiction over the church in Scotland but was forced to accept Canterbury's authority over York. He also worked on reconciling the Investiture Controversy between the king and the papacy over the right to appoint bishops until the controversy's resolution in 1107. Because of rumours, as a student of astrology, that he was a magician and a sorcerer, and also because of his unpopular attempts to reform his clergy, he was denied a burial inside York Minster"
#the_text = "Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet species by building a self-sustaining city on, Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed liquid-fuel launch vehicle to orbit the Earth."
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    #example_displacy_dependency("I love to drink coffee every morning")
    #example_displacy_entities(the_text)

    example_similarity("Apples and Oranges are my lovely fruits.")