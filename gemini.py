import pathlib
import textwrap
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv('google_api_key')

genai.configure(api_key=GOOGLE_API_KEY)
print(GOOGLE_API_KEY)
def gemini_response(content,query):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"""Based on the content please answer a query in a generalised manner and user the content as a reference to look of you can inprove your response or make user more satisfied asked , note that the context is from faq asked by the users in past and notice that there
                                       , Content : """ + content + "the user query is Query: "+ f"""{query}""")

    print(response.text)
    return response.text


# if __name__ =="__main__":
#    gemini_response(content= "why i am not able to login?   the reason bein yout email not being right", query = "i am not able to login?")
  