import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()


class ReportAnalyzer:
    def __init__(self):
        self.chat = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
        )
        # self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            model="text-embedding-3-large",
            chunk_size=1
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.str_parser = StrOutputParser()

    def process_pdf(self, pdf_file):
        pdf_path = f"temp/{pdf_file.name}"
        os.makedirs("temp", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())
        loader = PDFPlumberLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No text extracted from the PDF")
        text = "\n".join([doc.page_content for doc in docs])
        chunks = self.text_splitter.split_text(text)
        os.remove(pdf_path)
        return chunks, text

    def create_vectorstore(self, chunks, thread_id):
        os.makedirs("./chroma_db", exist_ok=True)
        documents = [Document(page_content=chunk, metadata={
                              "chunk_id": i}) for i, chunk in enumerate(chunks)]
        # vectorstore = Chroma.from_documents(
        #     documents=documents,
        #     collection_name=f"report_{thread_id}",
        #     embedding=self.embeddings,
        #     persist_directory="./chroma_db"
        # )
        vectorstore = FAISS.from_documents(
           documents,
           embedding=self.embeddings
       )
        return vectorstore

    def generate_summary(self, vectorstore):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        questions = [
            "What are the Scope 1, Scope 2, and Scope 3 emissions in the report?",
            "Does the report mention biodiversity initiatives or impacts?",
            "How many sustainability projects are completed and underway in the report?",
            "What are the total cost reductions from sustainability projects in the report?",
            "What is the reduction in water use from last year to this year, including amounts used this year, last year, and the delta?",
            "What are the specific sustainability projects mentioned, including their names, descriptions, cost reductions, and environmental impacts?"
        ]
        answers = []
        for question in questions:
            docs = retriever.invoke(question)
            context = "\n".join(
                [doc.page_content for doc in docs]) if docs else "No relevant data found."
            answer = self.answer_question(question, context)
            answers.append({"question": question, "answer": answer})
        return answers

    def generate_summarized_report(self, answers):
        context = "\n".join(
            [f"Q: {qa['question']}\nA: {qa['answer']}" for qa in answers])
        prompt = PromptTemplate(
            template="""Based on the sustainability report data, create a concise summarized report in Markdown. Include:
            - A brief overview.
            - A table for Scope 1, 2, and 3 emissions.
            - A table for water use (last year, this year, delta).
            - A list or table for sustainability projects (name, cost reduction, environmental impact).
            - Key insights or highlights.
            Data: {context}""",
            input_variables=["context"]
        )
        chain = prompt | self.chat | self.str_parser
        return chain.invoke({"context": context})

    def answer_question(self, query, context):
        prompt = PromptTemplate(
            template="""Answer the query based on the sustainability report data.
            Query: {query}
            Data: {context}
            Provide a concise, accurate response.""",
            input_variables=["query", "context"]
        )
        chain = prompt | self.chat | self.str_parser
        return chain.invoke({"query": query, "context": context})

    def generate_charts(self, answers, thread_id):
        chart_paths = []

        # Extract emissions data (assuming answers are in order)
        emissions_answer = answers[0]["answer"]
        try:
            # Parse emissions (example: "Scope 1: 50,000 tCO2e. Scope 2: 30,000 tCO2e. Scope 3: 200,000 tCO2e.")
            scope1 = float(emissions_answer.split("Scope 1: ")[
                           1].split(" tCO2e")[0].replace(",", ""))
            scope2 = float(emissions_answer.split("Scope 2: ")[
                           1].split(" tCO2e")[0].replace(",", ""))
            scope3 = float(emissions_answer.split("Scope 3: ")[
                           1].split(" tCO2e")[0].replace(",", ""))

            plt.figure(figsize=(6, 4))
            scopes = ["Scope 1", "Scope 2", "Scope 3"]
            values = [scope1, scope2, scope3]
            plt.bar(scopes, values, color=["#4CAF50", "#2196F3", "#FF9800"])
            plt.title("Greenhouse Gas Emissions")
            plt.ylabel("tCO2e")
            plt.grid(True, axis="y")
            emissions_path = f"temp/emissions_{thread_id}.png"
            plt.savefig(emissions_path, bbox_inches="tight")
            plt.close()
            chart_paths.append(emissions_path)
        except:
            pass  # Skip chart if parsing fails

        # Extract water use data
        water_answer = answers[4]["answer"]
        try:
            # Parse water use (example: "Last year: 1,000,000 gallons. This year: 900,000 gallons. Delta: 100,000 gallons (10% reduction).")
            last_year = float(water_answer.split("Last year: ")[
                              1].split(" gallons")[0].replace(",", ""))
            this_year = float(water_answer.split("This year: ")[
                              1].split(" gallons")[0].replace(",", ""))

            plt.figure(figsize=(6, 4))
            years = ["Last Year", "This Year"]
            values = [last_year, this_year]
            plt.bar(years, values, color=["#2196F3", "#4CAF50"])
            plt.title("Water Use Comparison")
            plt.ylabel("Gallons")
            plt.grid(True, axis="y")
            water_path = f"temp/water_{thread_id}.png"
            plt.savefig(water_path, bbox_inches="tight")
            plt.close()
            chart_paths.append(water_path)
        except:
            pass  # Skip chart if parsing fails

        return chart_paths
