from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from dotenv import load_dotenv
import urllib
import requests
import json
import os
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone
import pandas as pd

def get_csv_data():
    try:
        data = pd.read_csv("./faq_data.csv")
        return data
    except:
        return "data  is causing some issue"


def split_docs(text, chunk_size, chunk_overlap=50):
    print("size of chunk is ",chunk_size)
    print("*"*50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(text)
    print("length of docs is ",len(docs))
    return text_splitter.create_documents(docs),docs

load_dotenv()
HF_TOKEN = os.getenv("hf_token")

def persist_dir(chunks, csv_name):
    
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
    # print
    vecs = model.encode(chunks)
    print(vecs)
    print("Saving to db...")
    
    pc = Pinecone(api_key="6ee070dd-4af5-43a1-9abd-9f83426dda8c")
    index = pc.Index("shiva")
    data_list = []
    for i in range(len(chunks)):

        metadata = {}
        metadata["chunk_id"] = csv_name
        metadata["text"] = chunks[i]
        metadata["valid_chunk"] = True

        test_ins = {
                "id": csv_name, 
                "values":vecs[i], 
                "metadata": metadata
            }
        print("adding to data list")
        data_list.append(test_ins)
    index.upsert(vectors=data_list,namespace= "ns1")
    
    print("stored to database")
    
if __name__ == "__main__":
    data = get_csv_data()
    print(data.columns)
    for i in range(data.shape[0]):
        try:
            row = data.iloc[i]
        except Exception as e:
            print(e)
        content = f""" question:{row['Question']} and Amswer : {row["Answer"]} """
        docs,chunks = split_docs(content,500)
        
        persist_dir(chunks,str(i))
    