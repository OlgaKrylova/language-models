import numpy as np
import pandas as pd
import torch
import streamlit as st
# импортируем трансформеры
import transformers
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pretrained_weights = 'cointegrated/rubert-tiny-toxicity'
model = AutoModelForSequenceClassification.from_pretrained(pretrained_weights)
tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

def main():
    st.subheader('Тут можно определить токсичность сообщения :ru:')
    message = st.text_area(label='Сообщение тут', value = 'Да ты лошара!')
    result = st.button ('проверить')
    if result:
        result1 = text2toxicity(message, aggregate=False)
        result2 = np.array([result1[0] * (1 - result1[-1])])
        result1 = np.expand_dims(result1, axis=1)
        result2 = np.expand_dims(result2, axis=1)
        print(result1.shape, result2.shape)
        keys=['non-toxic',"insult", "obscenity", "threat", "dangerous", 'TEXT OK']
        values = np.concatenate((result1, result2), axis=0).squeeze(1)
        print(values)
        d={i:j for (i,j) in zip(keys, values)}
        print(d)
        df = pd.DataFrame(d, index=['probability'])
        st.write('Results',df)

if __name__ == '__main__':
         main()