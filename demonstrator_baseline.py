import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from IPython.display import Image, display, HTML
from IPython.core.display import display_html

from langchain.agents import (Agent, create_pandas_dataframe_agent, initialize_agent, Tool)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import (Prompt, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate)
from langchain.chains import (LLMChain, ConversationalRetrievalChain)
from langchain.document_loaders import DataFrameLoader
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


from openai.embeddings_utils import get_embedding, cosine_similarity
from chromadb.config import Settings
import chromadb
import tiktoken
import openai
from dotenv import load_dotenv

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

if __name__ == "__main__":
    
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    print("Loading data...")
    print("This will take a while depending on your hardware. We load only a subset of the data for demonstration purposes.")
    # df_art_embeddings_1 = pd.read_csv('data/article-embeddings_full_1.csv')
    # df_art_embeddings_1['ada_embedding'] = df_art_embeddings_1['ada_embedding'].apply(eval).apply(np.array)
    # df_art_embeddings_1

    # df_art_embeddings_2 = pd.read_csv('data/article-embeddings_full_2.csv')
    # df_art_embeddings_2['ada_embedding'] = df_art_embeddings_2['ada_embedding'].apply(eval).apply(np.array)
    # df_art_embeddings_2

    df_art_embeddings_3 = pd.read_csv('data/article-embeddings_full_3.csv')
    df_art_embeddings_3['ada_embedding'] = df_art_embeddings_3['ada_embedding'].apply(eval).apply(np.array)
    df_art_embeddings_3

    # frames = [df_art_embeddings_1, df_art_embeddings_2, df_art_embeddings_3]
    frames = [df_art_embeddings_3]

    df_art_embeddings_all = pd.concat(frames, ignore_index=True)

    search_term = input("Enter search term: ")
    # create vectors for search term
    search_term_vector = get_embedding(search_term, model='text-embedding-ada-002')

    df_art_embeddings_all["similarities"] = df_art_embeddings_all['ada_embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))

    df_recommendations = df_art_embeddings_all.sort_values("similarities", ascending=False).head(5)
  
    print(df_recommendations)