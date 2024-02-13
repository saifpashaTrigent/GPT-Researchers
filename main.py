from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import json
import streamlit as st
import time
from io import BytesIO
import math
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph,Spacer,Image
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


api_key = st.secrets["OPENAI_API_KEY"]



RESULTS_PER_QUESTION = 3

url = "https://www.google.com"

ddg_search = DuckDuckGoSearchAPIWrapper()
image_urls=[]

def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, write a blog for the following question and also include 
images links: 
> {question}
-----------
if the blog cannot be written using the text, imply summarize the text.
Include all factual information, numbers, stats,images links etc if available.""" 
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


def scrape_text(url: str):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            image_urls = [img["src"] for img in soup.find_all("img") if img.get("src")]
            return page_text,image_urls 
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"



scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
    text=lambda x: scrape_text(x["url"])[:10000]
) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()
) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()



SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings with images links in the following format: "
            '["query 1 with image", "query 2 with image", "query 3 with image"].',
        ),
    ]
)

search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = """You are an AI critical thinker research assistant.
                        Your sole purpose is to write well written, critically acclaimed,
                        objective and structured blogs with images links on given text."""  


RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, write a blog for the following question or topic: "{question}" 
\
The blog should focus on the answer to the question, should be well structured, 
informative, \
with facts and numbers and images links if available.
You should strive to write the blog as precise as you can using all relevant and 
necessary information provided.
Remember to add only 3 images links.
You MUST determine your own concrete and valid opinion based on the given information.
Do NOT deter to general and meaningless conclusions.
Do NOT forget to add images.
Write all used source urls at the end of the blog, and make sure to not add duplicated sources, 
but only one reference for each.


Please do your best, this is very important to my career."""  

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

# Remember to resize the image in a Height x Width of 400 × 400 px each.
def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-4-1106-preview") | StrOutputParser()




def generate_pdf_report(response_content):
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=60, leftMargin=60, topMargin=60, bottomMargin=60)
    styles = getSampleStyleSheet()
    styles['Normal'].fontSize = 12
    content = []

    lines = response_content.split('\n')
    for img_url in image_urls:
        try:

            img_response = requests.get(img_url, stream=True)
            
            if img_response.status_code == 200:
                img_data = BytesIO(img_response.content)
                img = Image(img_data)
                st.image(img)
                content.append(img)
                content.append(Spacer(1, 12))  
            else:
                print(f"Failed to download image: {img_url}")
        except Exception as e:
            print(f"Error downloading image: {e}")

    for line in lines:
        paragraph = Paragraph(line, styles['Normal'])
        content.append(paragraph)
        content.append(Spacer(1, 12))

    pdf.build(content)
    buffer.seek(0)
    return buffer


def main():

    favicon = Image.open("favicon.png")
    st.set_page_config(
        page_title="GenAI Demo | Trigent AXLR8 Labs",
        page_icon=favicon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar Logo
    logo_html = """
    <style>
        [data-testid="stSidebarNav"] {
            background-image: url(https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png);
            background-repeat: no-repeat;
            background-position: 20px 20px;
            background-size: 80%;
        }
    </style>
    """
    
    
    st.sidebar.markdown(logo_html, unsafe_allow_html=True)
    st.header("Create your blogs in 5 minutes with Gpt-Researchers 🔍")
    if api_key:
        success_message_html = """
        <span style='color:green; font-weight:bold;'>✅ Powering the Chatbot using Open AI's 
        <a href='https://platform.openai.com/docs/models/gpt-3-5' target='_blank'>gpt-3.5-turbo-0613 model</a>!</span>
        """

        # Display the success message with the link
        st.markdown(success_message_html, unsafe_allow_html=True)
        openai_api_key = api_key
    else:
        openai_api_key = st.text_input(
            'Enter your OPENAI_API_KEY: ', type='password')
        if not openai_api_key:
            st.warning('Please, enter your OPENAI_API_KEY', icon='⚠️')
        else:
            st.success('Ask Tech voice assistant about your software.', icon='👉')



    st.markdown("""
     ## ***Why GPT-Researchers❓***
    * GPT Researcher is an autonomous agent designed to perform comprehensive
                  online research on a variety of tasks..
    * Solutions that enable web search (such as ChatGPT + Web Plugin), only consider limited
                resources and content that in some cases result in superficial conclusions
                or biased answers.
    * Using only a selection of resources can create bias in determining the 
                right conclusions for research questions or tasks.
    """)

    a = st.text_input("Enter your Blog Title","Generative Ai vs Traditional Ai")
    if st.button("Generate"):
        with st.spinner("Generating blog..."):
            startTime = time.time()  
            response = chain.invoke({"question": a})
            endTime = time.time()  
            st.write(response)
            st.write("Time taken: ", math.floor(endTime - startTime)," seconds")
            generated_report_text  = response
            pdf_report = generate_pdf_report(generated_report_text)
            st.download_button(
                label="Download as PDF",
                data=pdf_report,
                file_name="blog.pdf",
                mime="application/pdf",
            )
    


    # Footer
    footer_html = """
    <div style="text-align: right; margin-right: 10%;">
        <p>
            Copyright © 2024, Trigent Software, Inc. All rights reserved. | 
            <a href="https://www.facebook.com/TrigentSoftware/" target="_blank">Facebook</a> |
            <a href="https://www.linkedin.com/company/trigent-software/" target="_blank">LinkedIn</a> |
            <a href="https://www.twitter.com/trigentsoftware/" target="_blank">Twitter</a> |
            <a href="https://www.youtube.com/channel/UCNhAbLhnkeVvV6MBFUZ8hOw" target="_blank">YouTube</a>
        </p>
    </div>
    """

    # Custom CSS to make the footer sticky
    footer_css = """
    <style>
    .footer {
        position: fixed;
        z-index: 1000;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
    }
    [data-testid="stSidebarNavItems"] {
        max-height: 100%!important;
    }
    </style>
    """

    # Combining the HTML and CSS
    footer = f"{footer_css}<div class='footer'>{footer_html}</div>"

    # Rendering the footer
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
