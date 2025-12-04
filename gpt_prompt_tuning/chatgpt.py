import openai
from openai import OpenAI, AzureOpenAI, AsyncAzureOpenAI
import json
from tqdm import tqdm

client = AzureOpenAI(
    api_key="",
    api_version="2024-02-01",
    azure_endpoint="https://api-ai-sandbox.princeton.edu/",
)

classes = ['playing guitar', 'bowling', 'playing saxophone', 'brushing teeth', 'playing basketball', 
'tying tie', 'skiing slalom', 'brushing hair', 'punching person (boxing)', 
'playing accordion', 'archery', 'catching or throwing frisbee', 'drinking', 
'reading book', 'eating ice cream', 'flying kite', 'sweeping floor', 'walking the dog', 
'skipping rope', 'clean and jerk', 'eating cake', 'catching or throwing baseball', 
'skiing (not slalom or crosscountry)', 'juggling soccer ball', 'deadlifting', 
'driving car', 'cleaning windows', 'shooting basketball', 'canoeing or kayaking', 
'surfing water', 'playing volleyball', 'opening bottle', 'playing piano', 'writing', 
'dribbling basketball', 'reading newspaper', 'playing violin', 'juggling balls', 'playing trumpet', 
'smoking', 'shooting goal (soccer)', 'hitting baseball', 'sword fighting', 'climbing ladder', 
'playing bass guitar', 'playing tennis', 'climbing a rope', 'golf driving', 'hurdling', 'dunking basketball']


result = {

}
for cls in tqdm(classes):
    print(cls)
    response = client.chat.completions.create(
            model='gpt-4o-mini',
            temperature=0.5, # temperature = how creative/random the model is in generating response - 0 to 1 with 1 being most creative
            max_tokens=1000, # max_tokens = token limit on context to send to the model
            top_p=0.5, # top_p = diversity of generated text by the model considering probability attached to token - 0 to 1 - ex. top_p of 0.1 = only tokens within the top 10% probability are considered
            messages=[
                {"role": "user", "content": f"Please give me five locations where this action could likely take place: {cls}. Answer in a single phrase, separated by comma. Don't start with any prepositions like in or at. The location should be singular, not plural. Words should all be lowercase, unless they are proper nouns."}, # user prompt
            ]
        )
    result[cls] = json.loads(response.json())
    print("\n"+response.choices[0].message.content)
 
json.dump(result, open("temp.json", "w"))