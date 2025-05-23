# -*- coding: utf-8 -*-
"""Colab_AND_LangChain+Qwen_4_SQL_Assistant.ipynb

Automatically generated by Colab.

### 1. Prepare Workspace
"""

!pip install transformers torch &> /dev/null

!pip install -U langchain &>/dev/null

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
from IPython.display import display, Markdown
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model
checkpoint = "Qwen/Qwen2.5-Coder-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Function to process the chat with mode selection for reasoning
def generate_response(system_message, user_message):
    conversation = f"{system_message.content}\nUser: {user_message.content}\nAI:"

    # Tokenize input and create attention mask
    tokens = tokenizer(conversation, return_tensors="pt")
    inputs = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Generate AI's response
    temperature = 0.1
    max_new_tokens = 300

    # produce a response from the AI model
    outputs = model.generate(inputs,
                             attention_mask=attention_mask,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature,
                             top_p=0.9, do_sample=True)

    # Decode and return the AIMessage
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = output_text.split('AI:')[-1].strip()
    return AIMessage(content=response_text)

"""### 2. Connection on SQLite Database"""

# connection to SQLite database
conn = sqlite3.connect('marketing.db')
cursor = conn.cursor()

"""### 3. Building tables"""

cursor.execute('''
DROP TABLE IF EXISTS customers;
''')
# Build customers table
cursor.execute('''
CREATE TABLE customers (
    customerID INTEGER PRIMARY KEY,
    city TEXT,
    occupation TEXT,
    age INTEGER
);
''')

cursor.execute('''
DROP TABLE IF EXISTS campaigns;
''')
# Build campaigns table
cursor.execute('''
CREATE TABLE campaigns (
    campaignID INTEGER PRIMARY KEY,
    product TEXT,
    start_date DATE,
    end_date DATE,
    budget NUMERIC,
    channel TEXT
);
''')

cursor.execute('''
DROP TABLE IF EXISTS sales;
''')
# Build sales table
cursor.execute('''
CREATE TABLE sales (
    saleID INTEGER PRIMARY KEY,
    customerID INTEGER,
    campaignID INTEGER,
    product TEXT,
    quantity INTEGER,
    sale_date DATE,
    saleamount NUMERIC
);
''')

conn.commit()

pd.read_sql_query("SELECT * FROM customers;", conn)

pd.read_sql_query("SELECT * FROM campaigns;", conn)

pd.read_sql_query("SELECT * FROM sales;", conn)

"""### Input Data"""

random.seed(0)

# Define customer data
customerIDs = list(range(1000, 1100))
cities = ["NEW YORK", "CHICAGO", "LOS ANGELES", "PHILADELPHIA"]
occupations = ["ENGINEER", "DOCTOR", "LAWYER", "TEACHER"]
ages = range(25, 60)

customers = []
for i in customerIDs:
    city = random.choice(cities)
    occupation = random.choice(occupations)
    age = random.choice(ages)
    customers.append((i, city, occupation, age))

# Insert data into the customers table
cursor.executemany('''
INSERT INTO customers (customerID, city, occupation, age)
VALUES (?, ?, ?, ?)
''', customers)
conn.commit()

pd.read_sql_query("SELECT *  FROM customers;", conn)

# Define campaigns data
campaignIDs = list(range(1, 101))
products = ["BIKE", "TENT", "KAYAK", "TREADMILL"]
channels = ["EMAILS", "TV", "SOCIAL NETWORK", "RADIO"]
budgets = range(1000, 50000, 1000)

campaigns = []
for i in campaignIDs:
    channel = random.choice(channels)
    product = random.choice(products)
    budget = random.choice(budgets)

    start_date = datetime(2020, 1, 1) + timedelta(
        days=random.randint(0, (datetime(2024, 12, 31) - datetime(2020, 1, 1)).days))
    end_date = start_date + timedelta(days=90)

    campaigns.append((i, product, start_date.date(), end_date.date(), budget, channel))

# Insert data into the campaigns table
cursor.executemany('''
INSERT INTO campaigns (campaignID, product, start_date, end_date, budget, channel)
VALUES (?, ?, ?, ?, ?, ?)
''', campaigns)
conn.commit()

pd.read_sql_query("SELECT *  FROM campaigns;", conn)

# Retrieve customer IDs
cursor.execute('SELECT customerID FROM customers')
customer_ids = [row[0] for row in cursor.fetchall()]

# Retrieve campaigns with their details
cursor.execute('SELECT campaignID, product, start_date, end_date FROM campaigns')
campaigns = cursor.fetchall()

# Parameters
num_sales = 100  # Total number of sales records to generate
quantity_range = (1, 10)  # Quantity per sale

# Define unit prices for products
unit_prices = {
    "BIKE": 300,
    "TENT": 150,
    "KAYAK": 400,
    "TREADMILL": 700
}

sales = []
sale_id_start = 1  # Starting ID for sales

for sale_id in range(sale_id_start, sale_id_start + num_sales):
    # Select a random campaign
    campaign = random.choice(campaigns)
    campaign_id, product, start_date_str, end_date_str = campaign

    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Generate a random sale date within the campaign period
    delta_days = (end_date - start_date).days
    random_days = random.randint(0, delta_days) if delta_days > 0 else 0
    sale_date = start_date + timedelta(days=random_days)

    # Select a random customer
    customer_id = random.choice(customer_ids)

    # Determine quantity and calculate sale amount
    quantity = random.randint(*quantity_range)
    sale_amount = quantity * unit_prices.get(product, 100)  # Default unit price if not specified

    # Create the sales record
    sales.append((
        sale_id,
        customer_id,
        campaign_id,
        product,
        quantity,
        sale_date.date(),
        sale_amount
    ))

# Insert data into the sales table
cursor.executemany('''
    INSERT INTO sales (saleID, customerID, campaignID, product, quantity, sale_date, saleamount)
    VALUES (?, ?, ?, ?, ?, ?, ?)
''', sales)

# Commit the transaction
conn.commit()

pd.read_sql_query("SELECT * FROM sales", conn)

"""### Prompt Engineering for join SQL query"""

# Define initial messages with prompt engineering
system_message = SystemMessage(content="You are a senior data analyst. Give an help in coding with SQL")
user_message = HumanMessage(content="""

  Please write an SQL query to connect the three tables according to the rules provided.

  Tables:

  1. customers (customerID INTEGER PRIMARY KEY, city TEXT, occupation TEXT, age INTEGER)
  2. campaigns (campaignID INTEGER PRIMARY KEY, product TEXT, start_date DATE, end_date DATE, budget NUMERIC, channel TEXT)
  3. sales (saleID INTEGER PRIMARY KEY, customerID INTEGER, campaignID INTEGER, product TEXT, quantity INTEGER, sale_date DATE, saleamount NUMERIC)

  Rules:
  Retrive all fields from the tables.

  SQL Query:
  """)

# Generate response
sql_query = generate_response(system_message, user_message)
print(sql_query.content)

"""### Run SQL join query


"""

pd.read_sql_query(
'''
SELECT
    c.customerID,
    c.city,
    c.occupation,
    c.age,
    ca.campaignID,
    ca.product AS campaign_product,
    ca.start_date,
    ca.end_date,
    ca.budget,
    ca.channel,
    s.saleID,
    s.quantity,
    s.sale_date,
    s.saleamount
FROM
    customers c
JOIN
    sales s ON c.customerID = s.customerID
JOIN
    campaigns ca ON s.campaignID = ca.campaignID;
''', conn)

"""### Prompt Engineering for group by SQL query"""

# Define initial messages with prompt engineering
system_message = SystemMessage(content="You are a senior data analyst. Give an help in coding with SQL")
user_message = HumanMessage(content="""

  Please write an SQL query according to the rules provided.

  Tables:

  1. customers (customerID INTEGER PRIMARY KEY, city TEXT, occupation TEXT, age INTEGER)
  2. campaigns (campaignID INTEGER PRIMARY KEY, product TEXT, start_date DATE, end_date DATE, budget NUMERIC, channel TEXT)
  3. sales (saleID INTEGER PRIMARY KEY, customerID INTEGER, campaignID INTEGER, product TEXT, quantity INTEGER, sale_date DATE, saleamount NUMERIC)

  Rules:
  Join all the tables. Group by total of saleamount per city and channel. The fields to show city, channel and sum of saleamount.

  SQL Query:
  """)

# Generate response
sql_query = generate_response(system_message, user_message)
print(sql_query.content)

"""### Run SQL group by query"""

pd.read_sql_query(
'''
SELECT
    c.city,
    ca.channel,
    SUM(s.saleamount) AS total_saleamount
FROM
    customers c
JOIN
    sales s ON c.customerID = s.customerID
JOIN
    campaigns ca ON s.campaignID = ca.campaignID
GROUP BY
    c.city, ca.channel;
''', conn)

"""### Prompt Engineering for group by and filter SQL query"""

# Define initial messages with prompt engineering
system_message = SystemMessage(content="You are a senior data analyst. Give an help in coding with SQL")
user_message = HumanMessage(content="""

  Please write an SQL query according to the rules provided.

  Tables:

  1. customers (customerID INTEGER PRIMARY KEY, city TEXT, occupation TEXT, age INTEGER)
  2. campaigns (campaignID INTEGER PRIMARY KEY, product TEXT, start_date DATE, end_date DATE, budget NUMERIC, channel TEXT)
  3. sales (saleID INTEGER PRIMARY KEY, customerID INTEGER, campaignID INTEGER, product TEXT, quantity INTEGER, sale_date DATE, saleamount NUMERIC)

  Rules:
  Join all the tables. Group by total of saleamount per city and channel. The fields to show are channel, city and total saleamount. Filter by "SOCIAL NETWORK" channel

  SQL Query:
  """)

# Generate response
sql_query = generate_response(system_message, user_message)
print(sql_query.content)

"""### Run group by and filter SQL query"""

pd.read_sql_query(
    '''
SELECT
    c.channel,
    cu.city,
    SUM(s.saleamount) AS total_saleamount
FROM
    customers cu
JOIN
    sales s ON cu.customerID = s.customerID
JOIN
    campaigns c ON s.campaignID = c.campaignID
WHERE
    c.channel = 'SOCIAL NETWORK'
GROUP BY
    c.channel, cu.city;
    ''',conn)

cursor.close()
conn.close()

"""### References"""

# https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
# https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
# https://python.langchain.com/api_reference/core/messages/langchain_core.messages.human.HumanMessage.html
# https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html
# https://python.langchain.com/api_reference/core/messages/langchain_core.messages.system.SystemMessage.html
