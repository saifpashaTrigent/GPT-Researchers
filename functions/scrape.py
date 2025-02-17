import requests
import streamlit as st
from bs4 import BeautifulSoup
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph,Spacer,Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

image_urls=[]

 
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



def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)