from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()
os.environ['LANGSMITH_PROJECT'] = 'Sequential Chain'

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 3 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
model2 = ChatOpenAI(model='gpt-4o', temperature=0.7)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    "run_name": "Sequential Chain",
    "tags": ["summarization", "report creation"],
    "metadata": {"model1": "gpt-4o-mini", "model2": "gpt-4o"},
}

result = chain.invoke({'topic': 'Water Scarcity in Africa'}, config=config)

print(result)
