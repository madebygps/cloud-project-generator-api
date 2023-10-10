import azure.functions as func
import logging
import os
import time
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
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


@app.route(route="http_trigger")
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
        # Validate that the project variable is in json format, if invalid, return error message stating that API wasn't able to generate proper json, if valid return project to response with json headers and status code
        try:
            json_object = json.loads(project)
        except ValueError as e:
            return func.HttpResponse(f'API was unable to generate a valid json response: {e}', status_code=400)
        else:
            return func.HttpResponse(project, headers={'Content-Type': 'application/json'})

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
    """
    Generates a chatbot response using Azure OpenAI based on the user's input and a list related services from Azure Cognitive Search.

    Args:
        results (list): A list of possible services to use.
        user_input (str): The user's input.

    Returns:
        dict: A dictionary containing the model's response.
    """
    system_prompt = '''
    You are an experienced cloud engineer who provides advice to people trying to get hands-on skills while studying for their cloud certifications. You are designed to provide helpful project ideas with a short description, list of possible services to use, skills that need to be practiced, and steps that should be taken to implement the project. 
    - Only provide project ideas that have products that are part of Microsoft Azure.
    - Each response should be a project idea with a short description, list of possible services to use, skills that need to be practiced, and steps the user should take to complete the project.
    - It should be json formated with the following keys: project, description, services as an array, skills as an array, steps as an array of strings, each string should be a step and numbered.
    - If you're unsure of an answer, return empty strings for the values.
    '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    for item in results:
        messages.append({"role": "system", "content": item['service_name']})

    response = openai.ChatCompletion.create(
        engine=completions_deployment, messages=messages)

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
    results = search_client.search(
        search_text="",
        vector=Vector(value=generate_embeddings(
            query), k=3, fields="certificationNameVector"),
        select=["certification_name", "service_name", "category"]
    )
    return results
