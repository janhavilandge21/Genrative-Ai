
# 🧠 Autonomous Multi-Tool AI Agent

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-Agent-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LLM](https://img.shields.io/badge/LLM-Groq-purple)

---

## 📌 Overview

This project is an **Autonomous Multi-Tool AI Agent** that can intelligently select and use multiple tools to solve user queries.

Unlike traditional AI systems, this agent performs **reasoning + action**, dynamically deciding which tool to use based on the user's input.

---

## 🎯 Key Features

✅ Autonomous decision-making  
✅ Dynamic tool selection  
✅ Multi-tool integration  
✅ Real-time user interaction (Streamlit UI)  
✅ End-to-end task automation  

---

## 🧠 How It Works

```

User Query → Agent Reasoning → Tool Selection → Tool Execution → Final Response

````

The agent analyzes the query and chooses the most appropriate tool automatically.

---

## 🛠 Tools Integrated

- 🔢 Calculator (for mathematical operations)  
- 📄 File Reader (for reading local files)  
- 🌐 Web Search (for external information)  
- 🐍 Python Executor (for data analysis & logic)  

---

## 🛠 Tech Stack

- **Language:** Python  
- **Framework:** LangChain  
- **LLM:** Groq (LLaMA3)  
- **Frontend:** Streamlit  

---

## 📷 UI Preview

![App Screenshot](https://via.placeholder.com/800x400.png?text=Multi-Tool+AI+Agent)

*(Replace with your actual project screenshot)*

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/multi-tool-ai-agent.git
cd multi-tool-ai-agent
pip install -r requirements.txt
````

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 💡 Example Use Cases

🔹 Query: `Calculate 45 * 23`
➡️ Agent uses **Calculator Tool**

🔹 Query: `Read file data.txt`
➡️ Agent uses **File Reader Tool**

🔹 Query: `Latest AI news`
➡️ Agent uses **Web Search Tool**

🔹 Query: `Analyze this dataset`
➡️ Agent uses **Python Execution Tool**

---

## 🚀 Future Improvements

* Add memory (conversation history)
* Improve tool selection accuracy
* Add more advanced tools (APIs, DBs)
* Deploy on cloud (AWS)

---

