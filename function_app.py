import azure.functions as func
import logging
import os
import json
import time
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (VectorizedQuery,
    VectorQuery,
    VectorFilterMode,    
)
import openai
from openai import AzureOpenAI

client = AzureOpenAI(api_key=os.environ['openai_api_key'],
api_version=os.environ['openai_api_version'], azure_endpoint=os.environ['openai_api_endpoint'])
from tenacity import retry, wait_random_exponential, stop_after_attempt

cog_search_endpoint = os.environ['cognitive_search_api_endpoint']
cog_search_key = os.environ['cognitive_search_api_key']
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base=os.environ['openai_api_endpoint'])'
# openai.api_base = os.environ['openai_api_endpoint']
embeddings_deployment = os.environ['openai_embeddings_deployment']
completions_deployment = os.environ['openai_completions_deployment']
prompt = ("PROVIDE EXACTLY ONE PROJECT IDEA IN JSON FORMAT FOR THE CERTIFICATION MENTIONED BELOW. PROJECT IDEA MUST: - INCLUDE A SHORT DESCRIPTION THAT DESCRIBES WHAT THE PROJECT IS ABOUT - INCLUDE A LIST OF POSSIBLE AZURE SERVICES TO USE TO BUILD THE PROJECT - INCLUDE A LIST SKILLS THAT WILL BE PRACTICED AS THE PROJECT IS BUILT - INCLUDE A LIST OF STEPS THE USER SHOULD TAKE TO COMPLETE THE PROJECT - BE JSON FORMATED WITH THE FOLLOWING KEYS: project, description, services AS AN ARRAY OF STRINGS, skills AS AN ARRAY OF STRINGS, steps AS AN ARRAY OF STRINGS - NOT INCLUDE ANY AZURE, MICROSOFT, AMAZON, GOOGLE OR ANY OTHER CLOUD PRODUCTS IN THE SKILLS LIST AND OR ARRAY BE HELPFUL AND DESCRIPTIVE YOU ARE AN EXPERIENCED CLOUD ENGINEER")
cog_search_cred = AzureKeyCredential(cog_search_key)
index_name = "project-generator-index"

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="http_trigger")
@app.function_name('http_trigger')
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    prompt = req.params.get('prompt')
    if not prompt:
        try:
            req_body = req.get_json()
        except ValueError:
            print("Caught ValueError for invalid JSON")
            return func.HttpResponse(
                body=json.dumps({'message': 'Invalid JSON request body and no prompt in the query string'}),
                status_code=400
            )
        else:
            prompt = req_body.get('prompt')
    if prompt:
        results_for_prompt = vector_search(prompt)
        completions_results = generate_completion(results_for_prompt, prompt)
        project = (completions_results.choices[0].message.content)
        try:
            project = json.loads(project)
        except ValueError:
            print("Caught ValueError for invalid JSON")
            return func.HttpResponse(
                body=json.dumps({'message': 'API was unable to generate proper JSON response'}),
                status_code=400
            )
        else:
            return func.HttpResponse(
                body=json.dumps(project),
                status_code=200,
                mimetype="application/json"
            )
        


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def generate_embeddings(text):
    '''
    Generate embeddings from string of text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    response = client.embeddings.create(input=[text], model=embeddings_deployment)
    embeddings = response.data[0].embedding
    time.sleep(0.5)  # rest period to avoid rate limiting on AOAI for free tier
    return embeddings


def generate_completion(results, user_input):
    """
    Generates a chatbot response using Azure OpenAI based on the user's input and a list related services from Azure Cognitive Search.

    Args:
        results (list): A list of possible services to use.
        user_input (str): The user's input.

    Returns:
        dict: A dictionary containing the model's response.
    """
    

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ]

    for item in results:
        messages.append({"role": "system", "content": item['service_name']})

    response = client.chat.completions.create(model=completions_deployment, messages=messages)

    return response


def vector_search(query):
    """
    Searches for documents in the index that are similar to the given query vector.

    Args:
        query (str): The query string to search for.

    Returns:
        SearchResult: The search result object containing the matching documents.
    """
    search_client = SearchClient(
        cog_search_endpoint, index_name, cog_search_cred)
    vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="certificationNameVector")
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["certification_name", "service_name", "category"]
    )
    return results
