from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from pydantic import BaseModel


def semantic_search_rag(query, vectorstore, llm_chain_model):
    num_chunks= 5
    retriever= vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
    # retriever = MultiQueryRetriever.from_llm(
    #             llm=llm_chain_model, 
    #             retriever=vectorstore.as_retriever(search_kwargs={"k": num_chunks})
    #             )
    
    template =''' Fill the blank in three to four words in given Statement from the following piece of context:
    context: {context} 
    statement: {statement}
    '''
    
    prompt = ChatPromptTemplate.from_template(template)
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "statement": RunnablePassthrough()}
    )

    output_parser= StrOutputParser()
    Results=list()
    
    for Query in query:
        context=  setup_and_retrieval.invoke(Query)
        prompt_answer= prompt.invoke({'context':context, 'statement': Query})
        model_answer= llm_chain_model.invoke(prompt_answer)
        response= output_parser.invoke(model_answer)
        Results.append(response)
    
    return Results
