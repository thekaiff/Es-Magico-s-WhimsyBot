# AI StoryBot (LangChain + RAG + TTS + Image Generation) 

![LangChain](https://img.shields.io/badge/LangChain-Framework-blue)
![TogetherAI](https://img.shields.io/badge/TogetherAI-LLM-orange)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-ff4b4b)
![License](https://img.shields.io/github/license/thekaiff/Es-Magico-s-WhimsyBot)
![Python](https://img.shields.io/badge/Python-3.10+-green)

This Bot is a voice-enabled, multilingual AI chatbot that delivers humorous answers to your questions using Retrieval-Augmented Generation (RAG) from classic literature. It supports text & voice input, speech output, cross-language queries, and even generates AI-powered images to bring stories to life.

> 💡 Submitted as part of the AI Internship Task for Es Magico AI Studio.


---


## ✨ Features

- 🧠 RAG-based question answering from 3 classic stories (PDFs)
- 🤖 Together AI LLM integration (DeepSeek-R1)
- 🧩 LangChain-powered prompt and retrieval system
- 🌍 Multilingual input/output with auto-detection (via Google Translate)
- 🎤 Audio input and 🗣️ Text-to-speech responses (gTTS)
- 🎨 AI-generated images (TogetherAI Flux)
- 📱 Intuitive Streamlit interface


---


## ⚙️ Installation

```bash
git clone https://github.com/yourusername/whimsybot.git
cd whimsybot
pip install -r requirements.txt
streamlit run app.py
```
🔑 Make sure to set your TOGETHER_API_KEY as an environment variable.


---


## 📁 Project Structure

```bash
Es-Magico-s-Whimsybot/
└── _pycache_/ 
├── chroma_db/              # Persisted vector store
└── demo_video/ 
├── pdfs/                   # Alice, Gulliver, Arabian Nights, Tech Scoping Write-Up
├── bot.py                  # Main Streamlit app
├── requirements.txt        # All dependencies
├── questions.txt           # Sample user queries
├── README.md               # This file 
```


---


## 📹 Video Demo

Watch the full walkthrough of WhimsyBot in action:
🎬 [Click to watch the demo video](./demo_video/demo.mp4)


---


## 🛠️ Tech Stack

- LangChain – For RAG & prompt chaining
- Together AI – LLM & image generation (DeepSeek-R1 & Flux)
- HuggingFace Embeddings – Semantic search
- gTTS / SpeechRecognition – Audio input/output
- deep-translator – Language detection and translation
- Streamlit – Interactive UI


---


## 📝 Assignment Scope

This project fulfills all core and bonus criteria:

- ✅ Knowledge training & retrieval logic
- ✅ Output tone control with custom prompts
- ✅ Image generation
- ✅ Easy model/config integration
- ✅ Audio input/output
- ✅ Multilingual support


---


## 👋 Contact
Feel free to reach out if you have feedback or ideas:
Kaif Anis Sayed
- 📧 kaifsdkpro2@gmail.com
- 🌐 [LinkedIn](https://www.linkedin.com/in/kaifsayed57/)


