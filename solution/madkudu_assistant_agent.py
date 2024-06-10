#Langchain
import os 
# Hello dear recruiter if you would like to test this you can get a open ai gpt for free if you create 
# a new account with a never seen number
os.environ["OPENAI_API_KEY"] = ""
from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


#Receives name of file
def load_and_process_data(name):

    #1. Loading data
    #using utf 8 since it has special characters
    loader = TextLoader(name, encoding='utf-8')
    documents = loader.load()

    #2. Split data in chunks
    text_splitter = RecursiveCharacterTextSplitter(
        #i chose 500 because the data is not that numerous
        # number of characters
        chunk_size = 500, 
        #How many chars are going to be overlapped in between each chunk
        chunk_overlap = 250,
        
        #length function to measure each chunk based on its character count.
        length_function = len,
        
        #includes index of character where the separation between each chunk is done
        add_start_index = True
        
        )

    chunks = text_splitter.split_documents(documents)

 
    # 3. Embedding model
    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    #4. Embed chunks and store them in a FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    #5. Create a retriever for the vector store
    data_retriever = vectorstore.as_retriever()
    return data_retriever

#big chain of little chains
def start_chain(retriever):

    #1. Instance our LLM
    llm = ChatOpenAI(
        #chose a cheap model to save credits
        model = "gpt-3.5-turbo-0125",
        #the closer to 0, the more factual the results are going to be
        temperature = 0.3
    )
    
   #2.1 Create Prompt containing Contextual knowledge retrived from the vectorstore
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "As a customer support agent named Matias for a company called Madkudu, respond precisely the user's questions based on the below context (If you do not find the answer mention the user to visit https://support.madkudu.com/hc/en-us) :\n\n{context}"),
        # this is the variable that should be received where the conversation is going to be stored
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    #2.2 Chain to answer with context
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    #3.1 Prompt that is going to be able to answer the LAST MESSAGE given the history
    last_msg_prompt = ChatPromptTemplate.from_messages([
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Use the above conversation to generate a search query to look up in order to get information relevant to the conversation.")
    ])

    #3.2 chain to answer past history and retrives similar vector from the database to the context
    retriever_chain = create_history_aware_retriever(llm, retriever, last_msg_prompt)

    #4. Conversational chain - join the two chains 
    #   stores conversation/queries context from the dcomuents     and answers with context(RAG)
    conversational_retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    return conversational_retrieval_chain

def talk_with_bot(conversation_chain, question, chat_history):
    answer = conversation_chain.invoke(
       { 
        "chat_history": chat_history,
        "input": question
        }
    )
    
    return answer["answer"]

if __name__ == "__main__":
    #intialize chain
    retriever = load_and_process_data('all.txt')
    assistant_chain = start_chain(retriever)
    
    #array to store the messages
    messages_historial = []
    
    start = True
    
    while True:
        
        if start:
            print("-------------------------Welcome----------------------------")
            print("Hello Duc and Ruben, my name is Matias and I am happy to apply to this internship. However, today I am going to be your customer service agent for Madkudu.")
            print("Please feel free to ask me any question related to the urls I used in my data extraction crawler. ")
            print("To finish the conversation type: bye ")
            start = False
        
        user_message = input("You:")
        if user_message.lower() == "bye":
            print("See you soon!")
            break
        
        answer = talk_with_bot(assistant_chain, user_message, messages_historial)
        messages_historial.append(HumanMessage(content=user_message))
        messages_historial.append(AIMessage(content=answer))
        print("Matias: ", answer) 
        
        