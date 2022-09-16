from catboost import CatBoostClassifier
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

stop_words = set(stopwords.words('english'))
vocab_list = pd.read_csv('vocab_list.csv')
vocab_to_int = vocab_list.set_index('Unnamed: 0').to_dict()['value']

def data_preprocessing(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text) # Remove HTML from text
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    text = [word for word in text.split() if word not in stop_words]
    text = [word for word in text if word in vocab_list['Unnamed: 0'].tolist()]
    text = ' '.join(text)
    return text

def padding(review_int, seq_len):
    if len(review_int) <= seq_len:
        zeros = list(np.zeros(seq_len - len(review_int)))
        features = zeros + review_int
    else:
         features = np.array(review_int[len(review_int)-seq_len :])
    return torch.Tensor(features).unsqueeze(0)

class sentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()
        
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)       
        self.dropout = nn.Dropout(0.3)   
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):

        batch_size = 1   
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        #stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # Dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        #sigmoid function
        sig_out = self.sigmoid(out)
        # reshape to be batch size first
        print(sig_out.shape)
        sig_out = sig_out.view(batch_size, -1)
        print(sig_out.shape)
        sig_out = sig_out[:,-1] # get last batch of labels
        print(sig_out.shape)
        
        return sig_out, hidden
    
    def init_hidden(self, batch_size=1):
        h0 = torch.zeros((self.n_layers,batch_size,self.hidden_dim))
        c0 = torch.zeros((self.n_layers,batch_size,self.hidden_dim))
        hidden = (h0,c0)
        return hidden

def classify_cb(review):
    from_file = CatBoostClassifier()
    from_file.load_model('models/cbc_model')
    y_pred = from_file.predict([review])
    p_pred = np.round(from_file.predict_proba([review]),2)
    answer_dict = {1: "'positive'", 0: "'negative'"}
    sentiment1 = answer_dict[y_pred] 
    text = 'CatBoost classifies this review as '+ sentiment1 + ' with ' + str(max(p_pred)) + ' probability '
    st.subheader(text)

def classify_rnn(review):
    review = data_preprocessing(review)
    print(review)
    review_int = [vocab_to_int[word] for word in review.split()]
    print(review_int)
    features = padding(review_int, seq_len = 400)
    print(features)
    rnn = sentimentLSTM(vocab_size=222610, output_size = 1, embedding_dim = 64, hidden_dim = 64, n_layers = 1)
    rnn.load_state_dict(torch.load('models/rnn_class_state_dict.pt'))
    hidden = rnn.init_hidden()
     #print(f'Prob of positive: {model(torch.Tensor(features).long(),(hidden,c))[0].item():.3f}')
    rnn.eval()
    answer_dict = {1: "'positive'", 0: "'negative'"}
    p_pred, _ = rnn(torch.Tensor(features).long(), hidden)
    sentiment2 = answer_dict[round(p_pred[0].item())] 
    text = 'RNN classifies this review as '+ sentiment2 + ' with ' + str(round(max(p_pred[0].item(), 1-p_pred[0].item()),2)) + ' probability '
    st.subheader(text)

def main():
    st.subheader('Тут с помощью CatBoost и RNN можно проверить отзыв на фильм. Работает на английском :uk:')
    review = st.text_area(label='Отзыв тут', value = 'Such an incredible movie!')
    result = st.button ('проверить')
    if result:
        st.write(classify_cb(review))
        st.write(classify_rnn(review))

if __name__ == '__main__':
         main()
