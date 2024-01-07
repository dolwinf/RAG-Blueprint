import chromadb
import yaml
import requests
import uuid

# Create embeddings
chroma_client = chromadb.PersistentClient(path="chromadb")
collection = chroma_client.get_or_create_collection(name="my_collection")


def embedding_function(text):

    url = 'https://api.openai.com/v1/embeddings'

    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }

    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['data']
    else:
        raise Exception(f"OpenAI API error: {response.text}")


with open('data.yaml', 'r') as file:
    try:
        file = yaml.safe_load(file)
        for text in file:
            doc_id = str(uuid.uuid4())
            # Do the embeddings here and save to Chroma DB
            res = embedding_function(text)
            collection.add(
                embeddings=[res[0]['embedding']],
                ids=doc_id
            )

    except yaml.YAMLError as exc:
        print(exc)
