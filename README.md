
# FoodHub AI-Powered Customer Support Chatbot

An AI-powered customer support chatbot developed for **FoodHub** to handle customer queries related to order status, delivery, payment, order items, and cancellations.

The chatbot uses an **LLM, SQL Agent, specialized tools, conversation history, and safety guardrails** to retrieve authorized order information and generate clear, polite, and customer-friendly responses.

---

## Problem Statement

Food delivery customers frequently raise queries about order status, delivery time, payment details, and cancellations. Manual handling of these queries can lead to longer response times and increased workload for customer support teams.

FoodHub aims to automate these common queries using an AI-powered chatbot connected to its structured order database.

---

## Project Objective

The objective is to build a chatbot that:

* Understands customer queries.
* Retrieves order information using an SQL Agent.
* Generates customer-friendly responses using an LLM.
* Maintains conversation history.
* Prevents unauthorized access to customer information.
* Provides cancellation guidance.
* Validates responses using output guardrails.
* Supports human escalation when required.

---

## Key Customer Queries

The chatbot handles queries such as:

* **Unauthorized access:** "I want to access the order details for every order."
* **Unresolved issue:** "I have raised the query multiple times. What is happening?"
* **Cancellation:** "I want to cancel my order."
* **Order tracking:** "Where is my order?"

---

## Dataset

The FoodHub order database contains the following fields:

| Column           | Description                 |
| ---------------- | --------------------------- |
| `order_id`       | Unique order identifier     |
| `cust_id`        | Customer identifier         |
| `order_time`     | Order placement timestamp   |
| `order_status`   | Current order status        |
| `payment_status` | Payment status              |
| `item_in_order`  | Items included in the order |
| `preparing_eta`  | Estimated preparation time  |
| `prepared_time`  | Actual preparation time     |
| `delivery_eta`   | Estimated delivery time     |
| `delivery_time`  | Actual delivery time        |

---

## Solution Architecture

**Customer Query → Input Guardrail → Chat Agent → SQL Agent → Order Query Tool → Answer Tool → Output Guardrail → Customer Response**

### Main Components

* **LLM:** Groq `qwen/qwen3.8-27b` for reasoning and response generation.
* **SQL Agent:** Retrieves authorized order information from the SQLite database.
* **Order Query Tool:** Interprets retrieved order information.
* **Answer Tool:** Converts the raw response into a polite and customer-friendly response.
* **Input Guardrail:** Blocks unauthorized, bulk, or sensitive requests.
* **Output Guardrail:** Validates generated responses before displaying them.
* **Conversation History:** Maintains previous customer and assistant messages using Streamlit session state.
* **Streamlit:** Provides the interactive chatbot interface.

---

## Key Features

* AI-powered customer support
* Order status and delivery tracking
* Payment and item-related queries
* SQL Agent database retrieval
* Conversation history
* Input and output guardrails
* Unauthorized access protection
* Order cancellation guidance
* Human escalation support
* Interactive Streamlit interface

---

## Technologies Used

* **Python**
* **Streamlit**
* **LangChain**
* **LangChain Experimental**
* **LangChain Groq**
* **Groq LLM**
* **SQLite**
* **SQLDatabase**
* **Pydantic**

---

## Project Files

```text
FoodHub/
├── app.py
├── customer_orders.db
├── requirements.txt
├── README.md
└── .gitignore
```

* **`app.py`** – Main Streamlit chatbot application.
* **`customer_orders.db`** – FoodHub order database.
* **`requirements.txt`** – Required Python dependencies.
* **`README.md`** – Project documentation.
* **`.gitignore`** – Files excluded from version control.

---

## Requirements

```text
streamlit
langchain==0.2.17
langchain-experimental==0.0.62
langchain-groq==0.1.6
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## API Configuration

The application uses the Groq API key through **Streamlit Secrets**.

```text
GROQ_API_KEY = "your_groq_api_key"
```

The API key should not be hard-coded or committed to the repository.

---

## Deployment

The Streamlit application can be launched using:

```bash
streamlit run app.py
```

The deployment requires:

* `app.py`
* `customer_orders.db`
* `requirements.txt`
* Configured `GROQ_API_KEY` in Streamlit Secrets

The application maintains conversation history using **Streamlit session state**.

---

## Google Colab Runtime

The project can be developed and tested in Google Colab.

The LLM is accessed through the **Groq API**, so local GPU acceleration is not required for LLM inference.

---

## Expected Outcome

The FoodHub chatbot automates common customer support queries by combining **LLM-based conversation, SQL database retrieval, specialized tools, conversation context, and safety guardrails** to provide accurate, safe, and customer-friendly responses.
