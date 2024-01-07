from create_index import embedding_function

import chromadb


# documents = []
chroma_client = chromadb.PersistentClient(path="chromadb")
collection = chroma_client.get_or_create_collection(name="my_collection")


# Get user input and convert to embedding to do a search agianst Chroma embeddings

def get_query(query):

    query_embeddings = embedding_function(query)
    collection = chroma_client.get_or_create_collection(name="my_collection")
    result = collection.query(
        query_embeddings=[query_embeddings[0]['embedding']],
        n_results=2
    )

    print(result)


get_query("Tell me something about how a Lion reproduces")
