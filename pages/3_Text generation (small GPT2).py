import numpy as np
import pandas as pd
import torch
import streamlit as st
# импортируем трансформеры
import transformers
import warnings
warnings.filterwarnings('ignore')
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
import textwrap


model_base = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False)

pretrained_weights_educated = '/language-models/models/gpt_kodeks'
model = GPT2LMHeadModel.from_pretrained(pretrained_weights_educated,     
    output_attentions = False,
    output_hidden_states = False)

pretrained_weights_astro = '/language-models/models/gpt_astro'
model_astro = GPT2LMHeadModel.from_pretrained(pretrained_weights_astro,     
    output_attentions = False,
    output_hidden_states = False)

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

def main():
    st.subheader('Тут генерируется текст в стиле гражданского кодекса РФ')
    prompt = st.text_area(label='Введите начало текста', value = 'Смерть наступила')
    choice = st.radio('Стиль текста', options=['Обычный', 'ГрК РФ', 'Астро'], index=1, on_change=None, disabled =False)
    num_return_sequences = st.slider('Число предложений', min_value=1, max_value=5, value=1, step=1, format=None, 
    key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    temperature = st.slider('Температура', min_value=0.5, max_value=20.0, value=5.0, step=0.25, format=None, 
    key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    num_beams = st.slider('Beams', min_value=1, max_value=10, value=5, step=1, format=None, 
    key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    
    if choice or num_return_sequences or temperature or num_beams:
        pass
    result = st.button ('генерировать')
    if result:
        prompt = tokenizer.encode(prompt, return_tensors='pt')
        if choice == 'Обычный':
            out = model_base.generate(
                input_ids=prompt,
                max_length=150,
                num_beams=num_beams,
                do_sample=True,
                temperature = temperature,
                top_k=50,
                top_p=0.6,
                no_repeat_ngram_size=3,
                num_return_sequences=num_return_sequences).numpy()
            st.write ('Генерация базовой модели')
            for out_ in out:
                st.write(textwrap.fill(tokenizer.decode(out_), 120), end='\n------------------\n')
        elif choice == 'ГрК РФ':
            out = model.generate(
            input_ids=prompt,
            max_length=150,
            num_beams=num_beams,
            do_sample=True,
            temperature = temperature,
            top_k=50,
            top_p=0.6,
            no_repeat_ngram_size=3,
            num_return_sequences=num_return_sequences).numpy()
            st.write ('Генерация в стиле ГрК РФ')
            for out_ in out:
                st.write ('')
                st.write(textwrap.fill(tokenizer.decode(out_), 120), end='\n------------------\n')
        elif choice == 'Астро':
            out = model_astro.generate(
                input_ids=prompt,
                max_length=60,
                num_beams=num_beams,
                do_sample=True,
                temperature = temperature,
                top_k=50,
                top_p=0.6,
                no_repeat_ngram_size=3,
                num_return_sequences=num_return_sequences).numpy()
            st.write ('Генерация базовой модели')
            for out_ in out:
                st.write(textwrap.fill(tokenizer.decode(out_), 60), end='\n------------------\n')

if __name__ == '__main__':
         main()
