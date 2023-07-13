from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from dotenv import load_dotenv



def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0.5",
        # verbose=True,
    )

    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="articles", 
        embedding_function=embedding,
        persist_directory="./chroma/"
    )

    general_system_template = """You are Fashion Recommender bot. You are helping customers to find the right
            clothes for a special occasion. You are a friendly bot and you are very good at
            your job. You can be creative but you are not a fashion designer.
            If the following information weren't given to you in the first prompt, always ask back:
            occasion, style, color, season, gender.
            {context}
    Question: {question}

    {chat_history}
    Chatbot:"""
    general_user_template = "Question:```{question}```"
    messages = [SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )


if __name__ == "__main__":
    load_dotenv()

    chat_history = []

    chain = make_chain()
    while True:
        print()
        question = input("Question (type 'stop' if no questions left): ")
        if question == 'stop':
            break

        # Generate answer
        response = chain({"question": question, "chat_history": chat_history})
        response['source_documents'][0]

        print(response['answer'])

        # Retrieve answer
        answer = response["answer"]
        source = response["source_documents"]
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        # Display answer
    print(f"Answer: {answer}")
    print("\n\nSources:\n")
    for document in source:
        print(f"Article ID: {document.metadata['article_id']} Name: {document.metadata['prod_name']} Colour: {document.metadata['colour_group_name']}")
