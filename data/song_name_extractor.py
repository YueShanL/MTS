from google import genai
from google.genai import types
import utils.config
import ast


def extract_name(titles):

    client = genai.Client(api_key=utils.config.getAPIValue('LLM'))
    contents = f'you are a music collection assistant, please extract the EXACT music name without adding any ' \
               f'authors\' name from the following video titles, considering carefully about the music name words and ' \
               f'formats. return in List format:{titles}'
    print(contents)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )
    print(response.text.replace('json', ''))
    return ast.literal_eval(response.text.replace('json', '').replace("`", ""))
