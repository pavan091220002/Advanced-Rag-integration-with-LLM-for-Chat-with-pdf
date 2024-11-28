from fastapi import FastAPI, File, UploadFile, Form,Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import PyPDF2

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(open('static/index.html').read())

@app.post("/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    with open("uploaded_file.pdf", "wb") as f:
        f.write(await pdf.read())
    loader = PyPDFLoader("uploaded_file.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    global pdf_texts
    pdf_texts = texts
    return {"message": "Pdf uploaded successfully"}

@app.post("/chat")
async def chat(pdfText: str = Form(...), input: str = Form(...)):
    global pdf_texts
    embeddings = OllamaEmbeddings(model="phi3")
    db = Chroma.from_documents(pdf_texts, embeddings)
    mod = Ollama(model="phi3")
    prompt=ChatPromptTemplate.from_template("""Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    I will tip you $1000 if the user finds the answer helpful. 
    <context>
    {context}
    </context>
    Question: {input}""")
    chain = create_stuff_documents_chain(mod, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, chain)
    response = retrieval_chain.invoke(input)
    return {response}
