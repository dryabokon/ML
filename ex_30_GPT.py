import time

import pandas as pd
import openai
import tools_DF
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
def open_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as in_file:
        key = in_file.read().split('\n')[0]
        return key
# ----------------------------------------------------------------------------------------------------------------------
def excel_to_pretty_text(filename_xls):
    df = pd.read_excel(filename_xls, engine='openpyxl')
    txt = tools_DF.prettify(df,showheader=True,showindex=False,tablefmt='psql',filename_out=None)
    #print(txt)
    return txt
# ----------------------------------------------------------------------------------------------------------------------
openai.api_key = open_file('openaiapikey_private_D.txt')
encoding='UTF8'
# ----------------------------------------------------------------------------------------------------------------------
def compose_prompt():

    propmt = 'original Mona liza'
    #propmt+= excel_to_pretty_text('D:\\Book1.xlsx')
    return propmt
# ----------------------------------------------------------------------------------------------------------------------
#prompt = "Hi there! Tell me please who is the best NBA player of all times?"
# prompt = "Please generate alternative 3 B-roll texts to augment the query below\n " \
#         "World cup captains want to wear rainbow armbands in Qatar\n"\

# ----------------------------------------------------------------------------------------------------------------------
def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0,stop=["<<END>>"]):
    prompt = prompt.encode(encoding=encoding, errors='ignore').decode()
    response = openai.Completion.create(engine=engine,prompt=prompt,temperature=temp,max_tokens=tokens,top_p=top_p,frequency_penalty=freq_pen,presence_penalty=pres_pen,stop=stop)
    text = response['choices'][0]['text'].strip()
    return text
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    prompt = compose_prompt()
    # print(prompt)
    # response = gpt3_completion(prompt)
    # print(response)

    response = openai.Image.create(prompt=prompt,n=1,size="256x256",)
    print(response["data"][0]["url"])

