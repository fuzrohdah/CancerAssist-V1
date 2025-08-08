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




# ========== QUERY ROUTING ==========

query = st.query_params.get("page", "home")

if query == "home":
    st.markdown(
        '<h2 style="color: #0077b6; font-size: 40px;">CancerAssist<sup>®</sup></h2>',
        unsafe_allow_html=True
    )
    st.markdown("""
        <p style='font-size: 18px;'>
        CancerAssist<sup>®</sup> is an AI-driven platform for breast cancer (BRCA) and melanoma analysis, featuring a knowledge-based chatbot (OncoBot), a CNN-powered skin lesion classifier (OncoScanSkin), and a BRCA subtype predictor (OncoTypeBC).
        </p>
    """, unsafe_allow_html=True)

    st.markdown("**Choose an AI Tool:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("OncoBot 🧠", use_container_width=True):
            st.query_params["page"] = "chat"
            st.rerun()
    with col2:
        if st.button("OncoScanSkin 🔎", use_container_width=True):
            st.query_params["page"] = "image"
            st.rerun()
    with col3:
        if st.button("OncoTypeBC 🧬", use_container_width=True):
            st.query_params["page"] = "brca"
            st.rerun()

    
    








elif query == "chat":
    if st.button("⬅️ Back to Home"):
        st.query_params.clear()
        st.rerun()

    #st.markdown(
        #'<h2 style="color: #222222; font-size: 30px;">RAG-based Informed ChatBot</h2>',
       # unsafe_allow_html=True
   # )
    st.markdown("""
    <h2 style='font-size: 30px; color: #222222;'>
    <span style='color:#222222;'>CancerAssist<sup>®</sup></span> OncoBot
    </h2>
    <p style='font-size: 18px;'>
    OncoBot combines MiniLM embeddings, FAISS-based document retrieval, and the Mistral language model to provide accurate, context-aware answers to breast cancer and melanoma questions. It draws from trusted medical sources to ensure reliability and clarity.
    </p>
    """, unsafe_allow_html=True)


    st.image(r"C:\Users\haris\OneDrive\Desktop\AI\Projects\Chatbot-RAG\final-pipeline.png", caption="RAG Pipeline", use_container_width=True)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(r"C:\Users\haris\OneDrive\Desktop\AI\Projects\Chatbot-RAG\faiss_index", embedding, allow_dangerous_deserialization=True)
    llm = ChatOllama(model="mistral")

    system_prompt = (
    "You are CancerAssist, a biomedical assistant specialized in explaining melanoma- and breast cancer-related medical information. Assume current year as 2025."
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
                polite_prompt = "You are CancerAssist. Reply politely to greetings or off-topic input."
                response = llm.invoke([
                    SystemMessage(content=polite_prompt),
                    HumanMessage(content=user_query)
                ])
                answer = response.content
                sources = []
            else:
                docs = vectorstore.similarity_search(user_query, k=8)
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
                    <div style='background-color:#fff; border-left:5px solid #ff6b6b; padding:0.75em; margin-bottom:0.5em; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.05); font-size:0.9em;'>
                        <strong>Source {i+1}:</strong> {doc.metadata.get('source', 'Unknown')}<br/>
                        {doc.page_content[:50]}...
                    </div>
                """, unsafe_allow_html=True)











elif query == "image":
    if st.button("⬅️ Back to Home"):
        st.query_params.clear()
        st.rerun()

    st.markdown("""
        <h2 style='font-size: 30px; color: #222222;'>CancerAssist<sup>®</sup> OncoScanSkin</h2>
        <p style='font-size: 18px;'>
        OncoScanSkin uses a convolutional neural network (CNN) trained on dermoscopic images to distinguish between benign and malignant skin lesions. It analyzes visual patterns in skin features to assist in early melanoma detection, offering real-time image-based classification.
        </p>     
    """, unsafe_allow_html=True)

    
    
    st.image(r"C:\Users\haris\OneDrive\Desktop\AI\Projects\Chatbot-RAG\Basic-CNN-Architecture.png", caption="Basic CNN Architecture", use_container_width=True)

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















elif query == "brca":
    if st.button("⬅️ Back to Home"):
        st.query_params.clear()
        st.rerun()

    # Now include your BRCA subtyping Streamlit code here:
    # ↓↓↓ Paste entire BRCA Classifier app code block from earlier here ↓↓↓

    # You can start with:
    #st.markdown("<h2 style='color:#1976D2;'>BRCA Subtype Classifier</h2>", unsafe_allow_html=True)
    import streamlit as st
    import pandas as pd
    import joblib
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load trained model
    model = joblib.load(r"C:\Users\haris\OneDrive\Desktop\AI\Projects\OncoTypeBC\rf_model.pkl")

    # App header
    
    st.markdown("""
    <h2 style='font-size: 30px; color: #222222;'>CancerAssist<sup>®</sup> OncoTypeBC</h2>
    <p style='font-size: 18px;'>
    An Integrated Multi-Omic Classifier for Breast Cancer Subtyping
    </p>
    <p style='font-size: 16px;'>
    Classifies BRCA subtypes: <b>Basal</b>, <b>Her2</b>, <b>LumA</b>, <b>LumB</b> using a Random Forest trained on <b>TCGA-BRCA multi-omic data (n=945)</b>.<br><br>
    Input: Excel file with 20 gene expressions, 5 mutation statuses, and 2 CNV amplifications (MYC, CCND1).<br>
    Format must exactly match training structure. Outputs predicted subtype + confidence.
    </p>
    <p style='font-size: 14px; color: #666;'>
    <i>Note: For research use only. Not clinically validated.</i>
    </p>
    """, unsafe_allow_html=True)
    
    # Upload Excel file
    st.markdown("<p style='font-size:16px; font-weight:bold;'>Upload Excel Sheet (20 + 5 + 2 features in order): </p>", unsafe_allow_html=True)    
    uploaded_file = st.file_uploader("Upload Excel file here", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            preds = model.predict(df)
            probs = model.predict_proba(df)
            confidences = probs.max(axis=1) * 100

            df['Predicted_Subtype'] = preds
            df['Confidence (%)'] = confidences.round(2)

            st.success("Prediction completed!")
            st.dataframe(df)

            # Show placeholder explanation
            #st.markdown("**Prediction Explanation:**")
            #st.info("LLM-based explanation is currently disabled in this version.")

            # Download button
            st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error! Cannot Predict!: {str(e)}")

    # Workflow image
    st.markdown("**Project Workflow:**")
    st.image(r"C:\Users\haris\OneDrive\Desktop\AI\Projects\OncoTypeBC\flowchart.png", caption="Fig. Project Workflow", use_container_width=True)

    # Pipeline
    st.markdown("""
    1. **Data Collection**: Retrieved gene expression data (in TPM - Transcripts Per Million), mutation data, and copy number variation (CNV) data for ~1,070 BRCA samples from TCGA via cBioPortal.
    2. **Initial Cleaning**: Removed duplicate entries, irrelevant columns, and non-primary tumor samples.
    3. **Feature Engineering**: Selected 20 expression genes using Mutual Information, plus 5 mutation and 2 amplification features.
    4. **Data Integration**: Merged datasets on sample ID with subtype labels.
    5. **Final Preprocessing**: Imputed missing values, fixed feature order, resulting in 945 clean samples.
    6. **ML Model**: Random Forest trained via RandomSearchCV — validation accuracy: **89%**.
    7. **Model Comparison**: SVC (87%), XGBoost (88%), ANN (85%) — RF was best in accuracy + interpretability.
    """)

    # Load and visualize training data
    merged_df = pd.read_csv(r"C:\Users\haris\OneDrive\Desktop\AI\Projects\OncoTypeBC\final_merged_df.csv", sep=",")
    st.subheader("Exploratory Overview of Training Data")

    st.markdown("**BRCA Subtype Distribution:**")
    fig1, ax1 = plt.subplots(figsize=(15, 10))
    sns.countplot(data=merged_df, x='subtype', palette="viridis", ax=ax1)
    ax1.set_ylabel("Sample Count")
    st.pyplot(fig1)

    st.markdown("**CNV Frequencies of Selected Genes:**")
    mutation_cnv_cols = [col for col in merged_df.columns if col.endswith("_m") or col.endswith("_amp")]
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    merged_df[mutation_cnv_cols].sum().plot(kind='bar', color='coral', ax=ax3)
    ax3.set_ylabel("Positive Samples")
    st.pyplot(fig3)

    # FAQ
    st.markdown("---")
    st.markdown("## FAQ", unsafe_allow_html=True)
    st.markdown("<div id='faq'></div>", unsafe_allow_html=True)

    st.markdown("**Q1. Why didn’t you include BRCA Normal samples?**")
    st.markdown("Due to very few available samples (~20), which is insufficient for learning stable patterns in machine learning models.")

    st.markdown("**Q2. Aren’t there simpler and cheaper histopathological methods to classify BRCA?**")
    st.markdown("Yes, IHC and pathology are standard and cost-effective. This tool is for integrative, research-grade molecular insights — not a clinical substitute.")


        # ...and the rest of your BRCA app code follows
