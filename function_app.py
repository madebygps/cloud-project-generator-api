import azure.functions as func
import logging
import os
import json
import datetime
import time
from azure.core.exceptions import AzureError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.models import Vector
from azure.search.documents.indexes.models import (
    IndexingSchedule,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchField,
    SearchFieldDataType,
    SearchableField,
    SemanticConfiguration,
    SimpleField,
    PrioritizedFields,
    SemanticField,
    SemanticSettings,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    SearchIndexerDataSourceConnection
)


import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt


cog_search_endpoint = os.environ['cognitive_search_api_endpoint']
cog_search_key = os.environ['cognitive_search_api_key']

openai.api_type = os.environ['openai_api_type']
openai.api_key = os.environ['openai_api_key']
openai.api_base = os.environ['openai_api_endpoint']
openai.api_version = os.environ['openai_api_version']
embeddings_deployment = os.environ['openai_embeddings_deployment']
completions_deployment = os.environ['openai_completions_deployment']
cog_search_cred = AzureKeyCredential(cog_search_key)
index_name = "project-generator-index"

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="project")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    prompt = req.params.get('prompt')
    if not prompt:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            prompt = req_body.get('prompt')

    if prompt:
        results_for_prompt = vector_search(prompt)
        completions_results = generate_completion(results_for_prompt, prompt)
        project = (completions_results['choices'][0]['message']['content'])
        response_data = {'project': project}
        return func.HttpResponse(json.dumps(response_data), headers={'Content-Type': 'application/json'})
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
            status_code=200
        )


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def generate_embeddings(text):
    '''
    Generate embeddings from string of text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    response = openai.Embedding.create(
        input=text, engine=embeddings_deployment)
    embeddings = response['data'][0]['embedding']
    time.sleep(0.5)  # rest period to avoid rate limiting on AOAI for free tier
    return embeddings


def generate_completion(results, user_input):
    system_prompt = '''
    You are an experienced cloud engineer who provides advice to people trying to get hands-on skills while studying for their cloud certifications. You are designed to provide helpful project ideas with a short description, list of possible services to use, and skills that need to be practiced.
    - Only provide project ideas that have products that are part of Microsoft Azure.
    - Each response should be a project idea with a short description, list of possible services to use, and skills that need to be practiced.
    - Write two lines of whitespace between each answer in the list.
    - If you're unsure of an answer, you can say "I don't know" or "I'm not sure" and recommend users search themselves.
    '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    for item in results:
        print(item)
        messages.append({"role": "system", "content": item['service_name']})

    response = openai.ChatCompletion.create(
        engine=completions_deployment, messages=messages)

    return response


def vector_search(query):
    search_client = SearchClient(
        cog_search_endpoint, index_name, cog_search_cred)
    results = search_client.search(
        search_text="",
        vector=Vector(value=generate_embeddings(
            query), k=3, fields="certificationNameVector"),
        select=["certification_name", "service_name", "category"]
    )
    return results