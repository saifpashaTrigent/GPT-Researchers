SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, write a blog for the following question and also include 
images links: 
> {question}
-----------
if the blog cannot be written using the text, imply summarize the text.
Include all factual information, numbers, stats,images links etc if available.""" 


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

WRITER_SYSTEM_PROMPT = """You are an AI critical thinker research assistant.
                        Your sole purpose is to write well written, critically acclaimed,
                        objective and structured blogs with images links on given text."""