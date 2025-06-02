import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Together
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile, os, re

STORY_PDFS = [
    "Alice_In_Wonderland.pdf",
    "Gullivers_Travels.pdf",
    "The_Arabian_Nights.pdf"
]
os.environ["TOGETHER_API_KEY"] = "tgp_v1_DJH3fG2Edrl160zXPg7z0Fwin9fKTv1E0XHEbSz9NpY"

@st.cache_resource
def get_vector_database(pdf_file_paths):
    chroma_directory = "./chroma_db"
    if os.path.exists(chroma_directory) and os.listdir(chroma_directory):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory=chroma_directory, embedding_function=embedding_model)
    all_documents = []
    for pdf_path in pdf_file_paths:
        all_documents.extend(PyPDFLoader(pdf_path).load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_documents)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(split_docs, embedding=embedding_model, persist_directory=chroma_directory)

def strip_llm_meta(answer_text):
    import re
    answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL | re.IGNORECASE).strip()
    answer_text = re.sub(r"</think>", "", answer_text, flags=re.IGNORECASE).strip()
    meta_lines = [
        r"^alright,? so.*", r"^let me (check|think|see).*", r"^oops, typo.*",
        r"^so, putting it all together.*", r"^here's what i can say.*",
        r"^i should probably.*", r"^maybe add something.*",
        r"^also, in the story.*", r"^the answer should be.*",
        r"^okay, so i need to.*", r"^hmm, maybe.*", r"^another angle.*"
    ]
    for pattern in meta_lines:
        answer_text = re.sub(pattern, "", answer_text, flags=re.IGNORECASE | re.MULTILINE).strip()
    paragraphs = [p.strip() for p in answer_text.split('\n') if p.strip()]
    if paragraphs:
        return paragraphs[-1]
    return answer_text.strip()

def is_generic_fallback(answer_text):
    fallback_responses = [
        "sorry, i don't know about that",
        "my brain is full of fairy tales",
        "ask me about alice, gulliver, or arabian nights"
    ]
    return any(phrase in answer_text.lower() for phrase in fallback_responses)

def text_to_speech(text, language_code):
    try:
        tts = gTTS(text=text, lang=language_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            return temp_audio.name
    except Exception:
        return None

def create_funny_image(prompt_text):
    import together, base64
    client = together.Together(api_key=os.environ["TOGETHER_API_KEY"])
    try:
        image_response = client.images.generate(
            prompt=prompt_text,
            model="black-forest-labs/FLUX.1-schnell",
            n=1, size="512x512", steps=8
        )
        if hasattr(image_response, "data") and image_response.data and hasattr(image_response.data[0], "url"):
            return image_response.data[0].url
        elif isinstance(image_response, dict) and 'data' in image_response:
            return image_response['data'][0]['url']
        elif isinstance(image_response, dict) and 'output' in image_response:
            return base64.b64decode(image_response['output'][0])
    except Exception as error:
        st.warning(f"Image generation failed: {error}")
    return None

# LangChain Setup :
vector_database = get_vector_database(STORY_PDFS)
story_llm = Together(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", temperature=0.7, max_tokens=512)
story_prompt_template = """
You are a humorous AI storyteller. ONLY use the provided context to answer the user's question in a funny tone.
Do NOT explain your reasoning. Do NOT include any 'thinking', 'let's think step by step', or meta-commentary.
If you don't know, reply: "Sorry, I Don't Know about that! My brain is full of fairy tales, not rocket science. Ask me about Alice, Gulliver, or Arabian Nights!"
<context>
{context}
</context>
User's question: {question}
Funny answer:
"""
story_prompt = PromptTemplate(input_variables=["context", "question"], template=story_prompt_template)
qa_chain = RetrievalQA.from_chain_type(
    llm=story_llm,
    retriever=vector_database.as_retriever(search_kwargs={"k":3}),
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": story_prompt}
)

# Streamlit UI :
st.title("🎙️ Funny AI StoryBot (RAG + LangChain + Together)")

if 'user_query' not in st.session_state:
    st.session_state['user_query'] = ""

user_query = st.text_input("Ask me anything:", value=st.session_state['user_query'])

import speech_recognition as sr
if st.button("🎤 Click to Speak"):
    with st.spinner("Listening..."):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 2.5
        try:
            with sr.Microphone() as mic_source:
                audio = recognizer.listen(mic_source, timeout=10, phrase_time_limit=15)
                recognized_text = recognizer.recognize_google(audio)
                st.session_state['user_query'] = recognized_text
                st.success(f"Recognized: {recognized_text}")
                st.rerun()
        except sr.UnknownValueError:
            st.error("Could not understand your speech.")
        except sr.RequestError as error:
            st.error(f"Speech recognition error: {error}")
        except Exception as error:
            st.error(f"Microphone error: {error}")

if user_query and user_query != st.session_state['user_query']:
    st.session_state['user_query'] = user_query

if st.session_state['user_query']:
    original_query = st.session_state['user_query']
    detected_language = "en"
    try:
        translated_query = GoogleTranslator(source='auto', target='en').translate(original_query)
        detected_language = GoogleTranslator().detect(original_query)
    except Exception:
        translated_query = original_query

    llm_result = qa_chain({"query": translated_query})

    # Get context chunks
    context_text = "\n".join([doc.page_content for doc in llm_result["source_documents"]]) if llm_result.get("source_documents") else ""

    # If no context, force fallback
    if not context_text.strip():
        bot_answer = (
            "I don't know... Sorry, I have no idea about that! "
            "My brain is full of fairy tales, not rocket science. "
            "Ask me about Alice, Gulliver, or Arabian Nights!"
        )
        fallback_mode = True
    else:
        bot_answer = strip_llm_meta(llm_result["result"])
        fallback_mode = is_generic_fallback(bot_answer)

    # Translate back if needed
    if detected_language != "en":
        try:
            bot_answer = GoogleTranslator(source='en', target=detected_language).translate(bot_answer)
        except Exception:
            pass

    st.markdown(f"**Bot says:** {bot_answer}")

    # Audio Output
    audio_file_path = text_to_speech(bot_answer, detected_language)
    if audio_file_path:
        st.audio(audio_file_path)

    # Image Generation (only if not fallback)
    if not fallback_mode:
        funny_image = create_funny_image(translated_query)
        if funny_image:
            st.image(funny_image, caption="Here's a funny image!", use_container_width=True)