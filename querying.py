from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
load_dotenv()
load_dotenv()
HF_TOKEN = os.getenv("hf_token")
def get_relevant_context(query,csv_name):
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
    # print
    vecs = model.encode(query).tolist()
    pc = Pinecone(api_key="6ee070dd-4af5-43a1-9abd-9f83426dda8c")
    index = pc.Index("shiva")
    result = index.query(
    vector=vecs,
    filter={
        "chunk_id": csv_name,
    },
    namespace="ns1",
    top_k=2,
    include_metadata = True
)

    context = ""
    for i in result.matches:
        context+=i.metadata["text"]
    # print(result)


    # print(context)
    return context


# if __name__ =="__main__":
#     get_relevant_context("Does this course also offer some ppo?","FAQ")
