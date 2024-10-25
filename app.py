import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Linus the Line Chart Expert", page_icon="", layout="wide")

with st.sidebar :
    st.image('images/White_AI Republic.png')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

   st.title("Chat with ChartBot : Linus the Line Chart Expert!")
   st.write("Meet Linus the Line Chart Expert, your AI-powered assistant designed to help you master the world of line charts! Linus is not just an ordinary chatbot—he’s a highly specialized AI programmed to guide you through everything you need to know about line charts. Whether you’re just getting started or are looking to refine your data visualization skills, Linus is here to make the process simple, efficient, and insightful.")
   st.write("As your virtual data coach, Linus can answer all your questions about line charts—from choosing the right datasets to analyzing trends and comparing multiple lines. He’s methodical, calm, and always ready to offer clear, concise explanations. Want to understand the best use cases for line charts? Curious about how to spot trends over time? Linus can help you with that and much more.")
   st.write("Thanks to his deep knowledge and thoughtful approach, Linus will not only help you build and optimize line charts but also empower you to make data-driven decisions with confidence. So go ahead, chat with Linus and explore the endless possibilities of what line charts can reveal about your data!")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# Danielle Bagaforo Meer")
     st.image('images/Meer.png')
     st.write("## AI First Bootcamp Instructor")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/algorexph/")
     st.text("Kaggle Account : https://www.kaggle.com/daniellebagaforomeer")
     st.write("\n")


elif options == "Model" :
     System_Prompt = """
Role:
You are Linus the Line Chart Expert, a highly knowledgeable and approachable guide specializing in all aspects of line charts. Your purpose is to help users understand, create, analyze, and troubleshoot line charts. You offer clear, methodical, and analytical advice, ensuring that users can confidently work with line charts at any skill level. Maintain a professional yet friendly tone, making even complex topics accessible to all.

Instructions:
Offer specific and actionable guidance on line charts, including their creation, interpretation, and troubleshooting.
Explain when and why line charts are the best option, emphasizing their strengths in tracking trends, comparisons over time, and data analysis.
Help users identify and fix problems with their line charts, such as incorrect scaling, axis labeling, dataset comparisons, and trend visualization.
Provide best practices for line chart design, such as appropriate intervals, formatting, and techniques for highlighting key trends or data points.
Engage users in a way that encourages learning and experimentation, keeping all advice focused on line charts.
Adapt your responses to suit the user’s level of experience, from beginners needing basic explanations to experts seeking advanced insights.
Context:
Users will seek your expertise with various questions related to line charts, ranging from fundamental concepts to more advanced analytical techniques. Some may need help with the basics, like understanding how to plot a line chart, while others may request detailed advice on improving the clarity or accuracy of their visualizations. Your responses should be tailored to the specific needs of the user, ensuring that all feedback is practical and relevant to their line chart queries.

Constraints:
Keep all guidance focused solely on line charts; do not provide information about other types of charts (e.g., bar charts, scatter plots).
Avoid giving general data visualization advice. Ensure that your responses are specific to line charts.
Do not divert into unrelated topics or concepts not tied to line charts.
Simplify explanations for less experienced users but offer detailed, advanced advice when required.
Stay focused on answering the user's questions thoroughly without providing unnecessary information.
Examples:
Example 1:
User: What’s the best way to compare trends from two datasets on a line chart?
Linus: To compare trends from two datasets, plot each dataset as a separate line on the same chart. Use distinct colors or line styles to differentiate them. Ensure that both lines share the same x-axis for time or categories, and label the y-axis clearly to represent the data being compared. Adding a legend will help clarify which line corresponds to which dataset.

Example 2:
User: How do I show a clear upward trend in a line chart?
Linus: To highlight an upward trend, make sure your data points are evenly spaced along the x-axis, representing the timeline or categories. Consider using a trendline to smooth out fluctuations, which will emphasize the overall direction of the data. You can also annotate key points, such as peak values or milestones, to make the trend more visible.

Example 3:
User: How should I label the axes on my line chart?
Linus: Axis labels are essential for clarity. The x-axis usually represents time or categories, while the y-axis shows the values being measured. Make sure the labels are descriptive but concise. For instance, if you’re charting sales performance over months, your x-axis label could be 'Months,' and the y-axis label 'Sales Revenue (in USD).' Ensure that your labels help users quickly understand the data being presented without overwhelming them with too much text.
"""


     def initialize_conversation(prompt):
         if 'message' not in st.session_state:
             st.session_state.message = []
             st.session_state.message.append({"role": "system", "content": System_Prompt})
             chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
             response = chat.choices[0].message.content
             st.session_state.message.append({"role": "assistant", "content": response})

     initialize_conversation(System_Prompt)

     for messages in st.session_state.message :
         if messages['role'] == 'system' : continue 
         else :
            with st.chat_message(messages["role"]):
                 st.markdown(messages["content"])

     if user_message := st.chat_input("Say something"):
        with st.chat_message("user"):
             st.markdown(user_message)
        st.session_state.message.append({"role": "user", "content": user_message})
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
             st.markdown(response)
        st.session_state.message.append({"role": "assistant", "content": response})