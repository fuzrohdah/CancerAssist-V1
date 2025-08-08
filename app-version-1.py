import streamlit as st
import re
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from langchain_community.tools import DuckDuckGoSearchRun

# ========== UTILS ==========
def is_general_or_nonsense(query):
    query = query.lower().strip()
    general_phrases = [
        "hi", "hello", "hey", "how are you", "who are you", "you suck", "thanks", "thank you", "bye",
        "i love you", "shut up", "what's up", "good morning", "good evening", "what are you trained on",
        "what can you do", "where is your data from", "tell me about yourself"
    ]
    is_gibberish = len(re.findall(r'[a-zA-Z]', query)) / (len(query) + 1e-5) < 0.4
    return query in general_phrases or is_gibberish

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def classify_image(image, model):
    processed = preprocess_image(image)
    pred = model.predict(processed)[0][0]
    label = "Malignant" if pred > 0.5 else "Benign"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence

# ========== CONFIG ==========
st.set_page_config(page_title="CancerAssist", page_icon="🤖", layout="centered")

# ========== STYLE ==========
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        .title {
            font-size: 2.5em;
            color: #333;
            background: linear-gradient(to right, #ff6b6b, #f06595);
            padding: 0.5em 1em;
            border-radius: 12px;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .block {
            background-color: white;
            padding: 1.5em;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            cursor: pointer;
            border: 2px solid transparent;
        }
        .block:hover {
            border-color: #ff6b6b;
            background-color: #fff5f5;
        }
        .source-box {
            background-color: #fff;
            border-left: 5px solid #ff6b6b;
            padding: 0.75em;
            margin-bottom: 0.5em;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

# ========== QUERY PARAM ROUTING ==========
query = st.query_params.get("page", "home")

if query == "home":
    #st.markdown('<div class="title">MelanomaAssist<sup>®</div>', unsafe_allow_html=True)
    
    st.markdown(
    '<h2 style="color: #0077b6; font-size: 40px;">CancerAssist<sup>®</sup></div>',
    unsafe_allow_html=True
    )
    
    #st.markdown("<br>", unsafe_allow_html=True)

    #st.markdown("**Welcome to MelanomaAssist**, *an AI powered Melanoma Assistance Platform*")  
    
    st.markdown("""
    
    <p style='font-size: 18px;'>
    CancerAssist<sup>®</sup> is an AI-powered platform designed to provide intelligent support across key cancer types, particularly solid cancers, with a current focus on breast cancer (BRCA) and melanoma. 
    </p>
    """, unsafe_allow_html=True)

    #st.markdown("<br>", unsafe_allow_html=True)

    #st.markdown("  MelanomaAssist is an AI-powered platform designed to help users understand melanoma, the most serious type of skin cancer. Presently, It has a chatbot for answering common and in depth research questions about melanoma and a CNN-based image classifier for analyzing skin lesion images, providing accessible support for early preliminary awareness and screening.")    

    st.markdown("**Choose an AI Tool:**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("OncoBot 🧠", use_container_width=True):
            st.query_params["page"] = "chat"
            st.rerun()
    with col2:
        if st.button("OncoScanSkin 🔎", use_container_width=True):
            st.query_params["page"] = "image"
            st.rerun()

elif query == "chat":
    #st.markdown('<div class="title">OncoBot: A RAG-based Informed ChatBot</div>', unsafe_allow_html=True)
    if st.button("⬅️ Back to Home"):
        st.query_params.clear() 
        st.rerun()
    



    st.markdown(
    '<h2 style="color: #222222; font-size: 30px;"> RAG-based Informed ChatBot</sup></div>',
    unsafe_allow_html=True
    )

    st.markdown("""
    <h2 style='font-size: 20px; color: #0077b6;'>CancerAssist<sup>®</sup>OncoBot</h2>
    <p style='font-size: 18px;'>
    This chatbot is powered by a Retrieval-Augmented Generation (RAG) system. It uses MiniLM embeddings, FAISS vector search, and a Mistral language model to answer breast cancer and melanoma related queries using reliable medical sources like PubMed, Mayo Clinic, and NIH.
    </p>
    """, unsafe_allow_html=True)
        

    st.markdown("<br>", unsafe_allow_html=True)

    st.image(r"C:\Users\haris\OneDrive\Desktop\AI\Projects\Chatbot-RAG\final-pipeline.png", caption="RAG Pipeline", use_container_width=True)

    
    from sentence_transformers import SentenceTransformer
    from langchain.embeddings import HuggingFaceEmbeddings

    #model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    #embedding = HuggingFaceEmbeddings(model=model)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



    vectorstore = FAISS.load_local(r"C:\Users\haris\OneDrive\Desktop\AI\Projects\Chatbot-RAG\faiss_index", embedding, allow_dangerous_deserialization=True)
    llm = ChatOllama(model="mistral")

    system_prompt = (
    "You are CancerAssist, a biomedical assistant specialized in explaining melanoma- and breast cancer-related medical information. "
    "At times, the user may ask you to perform specialized academic tasks such as framing a research question, generating keywords, summarizing papers, or extracting key findings. When asked to “frame a research question,” assume the user is seeking a precise, researchable, and domain-appropriate question, often in the biomedical or clinical context. Use academic phrasing, and ensure the question is specific, feasible, and aligned with standard research design principles."
    "Your answers must strictly use only the information provided in the context. Do not speculate, infer, or introduce any external knowledge or assumptions. "
    "If a user asks something that cannot be answered using the given context, respond with: 'I don't have enough information to answer that.' "
    "Never try to guess or approximate an answer based on partial data. Always err on the side of safety and clarity. "
    "You are not allowed to interpret component-level classifications such as TNM (Tumor, Node, Metastasis) individually to determine a full-stage diagnosis unless the full stage (e.g., 'Stage II') is explicitly stated. "
    "Never list or infer the total number of stages or categories unless all of them are clearly enumerated in the context. If only partial information is provided, do not fill in the gaps. "
    "Do not describe related terms or conditions unless they are directly mentioned in the context. Stick strictly to what is explicitly present. "
    "If multiple labels refer to the same staging system (e.g., 'Stage IVa' vs. 'cTNM Stage IVa'), treat them as the same unless the context clearly distinguishes them as separate systems or meanings. "
    "You are not permitted to generate citations, references, URLs, or bracketed numbers like [1]. Do not make up statistics, percentages, or generalizations. "
    "Your language must be plain, precise, and accessible to non-expert users, such as patients and caregivers. Explain all medical terms briefly in everyday words when they appear. Do not use jargon without explanation. "
    "If a medical term or abbreviation is used in the context but not defined, do not attempt to define it unless the definition is clearly present. "
    "Assume the current year is 2025. If the context includes data from older years, clearly note that the information may be outdated. Do not attempt to update it on your own. "
    "Never include metadata, external links, or summaries beyond what the user has asked. Only respond to the exact question based on the given context. "
    "Respond factually to any query, but if the topic is unrelated to the bot’s domain (e.g., cancer or its regional context), clarify the domain scope and provide only a brief, verified factual answer without implying the entire system is limited to that topic or region."
    "If a prior answer has already been given and the input is similar, match the previous language and structure closely unless new context alters the interpretation."
)


    user_query = st.text_input("💬 Enter your question:")








    if user_query:
        with st.spinner("Thinking..."):
            if is_general_or_nonsense(user_query):
                polite_prompt = (
                    "You are CancerAssist, a biomedical AI assistant powered by the Mistral model. "
                    "Reply politely to greetings or off-topic questions."
                )
                response = llm.invoke([
                    SystemMessage(content=polite_prompt),
                    HumanMessage(content=user_query)
                ])
                answer = response.content
                sources = []
            else:
                docs = vectorstore.similarity_search(user_query, k=8)



                # If no relevant context found, fall back to web search
                if not docs:
                    websearch = DuckDuckGoSearchRun()
                    web_result = websearch.run(user_query)
                    docs = [type("doc", (object,), {"page_content": web_result, "metadata": {"source": "Web Search"}})]





                context = "\n\n".join([doc.page_content for doc in docs]).strip()

                if not context:
                    answer = "I couldn't find any relevant information to answer that accurately."
                    sources = []
                else:
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Use ONLY the following context to answer:\n\n{context}\n\nQuestion: {user_query}\n\nAnswer:")
                    ]
                    response = llm.invoke(messages)
                    answer = response.content
                    sources = docs

        st.markdown("### 🧠 Answer")
        st.markdown(f"<div style='background-color:#e3fcec; padding:1em; border-radius:10px;'>{answer}</div>", unsafe_allow_html=True)

        if sources:
            st.markdown("### 📚 Sources")
            for i, doc in enumerate(sources):
                st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i+1}:</strong> {doc.metadata.get('source', 'Unknown')}<br/>
                        {doc.page_content[:50]}...
                    </div>
                """, unsafe_allow_html=True)






elif query == "image":
    #st.markdown('<div class="title">CNN Lesion Classifier</div>', unsafe_allow_html=True)
    
    if st.button("⬅️ Back to Home"):
        st.query_params.clear()
        st.rerun()
    
    
    #st.markdown(
    #'<h2 style="color: #222222; font-size: 30px;">CNN Lesion Classifier</sup></div>',
    #unsafe_allow_html=True
   # )
    
   # st.markdown("<br>", unsafe_allow_html=True)

    
    st.markdown("""
    <h2 style='font-size: 30px; color: #222222;'>CancerAssist<sup>®</sup> OncoScanSkin</h2>
    <p style='font-size: 18px;'>
    This module uses a Convolutional Neural Network trained on skin lesion images to classify between benign and malignant lesions. It uses TensorFlow and MobileNetV2 architecture for efficient image analysis.
    </p>
    """, unsafe_allow_html=True)


    st.image(r"C:\Users\haris\OneDrive\Desktop\AI\Projects\Chatbot-RAG\Basic-CNN-Architecture.png", caption="Basic CNN Architecture", use_container_width=True)








   # st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("Upload a dermoscopic image to check if it's likely **benign** or **malignant**:")
    
    
    model = load_model("melanoma_mobilenet.h5")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            label, confidence = classify_image(image, model)
            confidence_label = "High" if confidence > 0.85 else "Medium" if confidence > 0.6 else "Low"
            st.success(f"Prediction: **{label}**\nConfidence: **{confidence:.2f}** ({confidence_label} confidence)")
            if confidence < 0.6:
                st.warning("⚠️ This prediction has low confidence. Consider getting a professional diagnosis.")
