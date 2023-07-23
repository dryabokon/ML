import cv2
import pandas as pd
from PIL import Image
import requests
# ----------------------------------------------------------------------------------------------------------------------
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
# ----------------------------------------------------------------------------------------------------------------------
model_id = "openai/clip-vit-base-patch32"
# ----------------------------------------------------------------------------------------------------------------------
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
# ----------------------------------------------------------------------------------------------------------------------
image1 = cv2.imread('./data/ex_caption/image05.jpg')
image2 = cv2.imread('./data/ex_caption/image06.jpg')
images = [image1,image2]
# ----------------------------------------------------------------------------------------------------------------------
text1 = "horse"
text2 = "two cats"
text3 = 'cat'
text4 = 'dog'
text5 = 'witch'
text6 = 'woman'
texts = [text1,text2,text3,text4,text5,text6]
# ----------------------------------------------------------------------------------------------------------------------
def ex_text_features(text):

    inputs = tokenizer(text, return_tensors="pt")
    text_emb = model.get_text_features(**inputs)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_image_text_similarity(image,text):

    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    similarity = outputs.logits_per_image.softmax(dim=1).cpu().detach().numpy()

    # inputs = tokenizer(text, return_tensors="pt")
    # text_emb = model.get_text_features(**inputs)

    #similarity = similarity[0, 0]
    print(similarity)
    return similarity
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # inputs = tokenizer(texts, return_tensors="pt")
    # image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    ex_image_text_similarity(images,texts)