from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from vectorstsore import vectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from lcel_rag import semantic_search_rag
from config import *
from dotenv import load_dotenv
from langchain_community.llms import Ollama
load_dotenv()

embed_fn = HuggingFaceEmbeddings(model_name=embed_model)


''' Function to load document from Folder'''
def document_loader(file_path):  
    loader=PyMuPDFLoader(file_path)
    Texts=loader.load()
    return Texts

''' Function to split pages into chunks'''
def slpitter(Pages):
    text_splitter=RecursiveCharacterTextSplitter(
            separators=["\n","."],
            chunk_size= Chunk_size,
            chunk_overlap= Chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    Chunks_list=[]
    for page in Pages:
        File_name=page.metadata['source'].split("\\")[-1]
        page_number=page.metadata['page']
        chunks=text_splitter.split_text(page.page_content)
        for chunk in chunks:
            meta_data={'source':File_name,'page_number':page_number}
            doc_string=Document(page_content=chunk,metadata=meta_data)
            Chunks_list.append(doc_string)
    return Chunks_list

def get_llm_model(model_name):
    llm = Ollama(model=model_name)
    return llm

def create_vector_store(chunks):
    init_vectorstore = vectorStore(is_load, index_name)
    vector_db = init_vectorstore.create_vectorstore_qdrant(embed_fn, chunks)
    return vector_db

'''Functio to store output in text file'''
def write_list_to_file(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data_list:
            file.write(item + '\n\n')


def main():
    '''calling Document Loader'''
    Pages=document_loader(Document_path)

    '''calling Splitter'''
    Chunks=slpitter(Pages)

    queries= ["The Arabs wanted to kill Hazrat Muhammad because _________ ",
    "Cruel practices of Arabs stopped because _________ ",
    "The old woman threw rubbish on Hazrat Muhammad because _________ ",
    "Hazrat Muhammad visited the old woman because _________ "]
    
    ''' Crating Vector DB '''
    vector_db = create_vector_store(Chunks)
    
    ''' Get LLM model '''
    llm = get_llm_model(model_name="phi3")

    ''' Retreive Data '''
    answers = semantic_search_rag(queries, vector_db, llm)

    ''' store output in text file '''
    write_list_to_file(answers, "Output.txt")

if __name__ == "__main__":
    main()
