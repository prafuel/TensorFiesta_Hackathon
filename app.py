# for fetch image
import requests
from bs4 import BeautifulSoup

import random

import pandas as pd
import numpy as np


df = pd.read_csv("./datasets/clean_reviews.csv")

import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.environ['GEMINI_API_TOKEN'])

model = genai.GenerativeModel('gemini-pro')

def get_insight(df:pd.DataFrame, x_label:str, y_label:str):
    prompt = f'''{np.array(df)} based on the provided array, your task is to analys it in term of business point of view, by considering x_lable={x_label} which is feature1, and y_label={y_label} which is number of sold product at that price, give me insight on it in following format: 1] <insight point>, give at least 4 point of analysis and next 2-3 points for solution'''
    result = model.generate_content(prompt)
    return result.text.replace("*","")


# dataframe for sentimental analysis 
senti = pd.read_csv("./datasets/sentiment_reviews.csv")
good = senti[senti['sentiment'] == 1]
bad = senti[senti['sentiment'] == 0]

import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.style.use('ggplot')

import plotly.express as px

import streamlit as st
st.set_page_config(layout='wide')

from bs4 import BeautifulSoup
import requests

def product_img_url(url: str):
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'lxml')
    img = soup.findAll('img')[4]
    # print(img['src'])
    return img['src']

def get_line(df: pd.DataFrame):
    col = df.columns
    figure = px.line(df, x = col[0], y=col[1])
    return figure

def get_bar(df: pd.DataFrame):
    col = df.columns
    figure = px.bar(df, x=col[0], y=col[1])
    return figure

def get_pie(df: pd.DataFrame):
    col = df.columns
    figure = px.pie(df, names=col[0], values=col[1])
    return figure


def print_cols(hero):
    index = 0
    for col in st.columns(spec=hero.shape[0]):
        with col:
            name, url, sku, price = np.array(df[df['sku'] == hero[index]][['product_name', 'url','sku', 'price']].drop_duplicates()).reshape(-1)
            
            res = requests.get(url)
            soup = BeautifulSoup(res.content, 'lxml')
            img = soup.findAll('img')[5]['src']
            # st.text(img)
            st.image(img,caption=name+" : "+str(price)+"rs", use_column_width=True)
            st.text(f"{sku}")
        index += 1

# st.title("Text Analysis")
radio = st.sidebar.radio(
    "Select Option",options=['Overall','Product_category', 'States', 'Pack_size','Products']
)

arr = ['price','product_category','states','pack_size']

# Sidebar Radio
if radio == "Overall":
    # working here to add most popular product from whole dataset
    # into the overall section, growth by date, year, month

    st.header("Overall Graph Analysis")

    op1 = st.selectbox("Select from following", options=arr)

    st.header(op1.capitalize())
    g = df[op1].value_counts().reset_index()
    main = g.copy()
    if op1 == 'price':
        g = df[op1].value_counts().reset_index().sort_values(by='price', ascending=True)

    main['percentage'] = g['count'].apply(lambda x : x*100/df.shape[0])
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(main)
    with col2:
        pass
    
    col1, col2 = st.columns(spec=2)
    with col1:
        st.plotly_chart(get_line(g))
    with col2:
        st.plotly_chart(get_pie(g))


    st.title("Sentiment Reviews")

    col1, col2 = st.columns(spec=2)

    with col1:
        st.text("Positive Review vs Negative Review ")
        if op1 == "price":
            fig1 = get_line(good[op1].value_counts().reset_index().sort_values(by='price', ascending=True))
            fig1.update_traces(line=dict(color = 'rgb(255,255,255)'))
            
            fig2 = get_line(bad[op1].value_counts().reset_index().sort_values(by='price', ascending=True))
            st.plotly_chart(go.Figure(data=fig1.data + fig2.data))
        else:
            fig1 = get_line(good[op1].value_counts().reset_index())
            fig1.update_traces(line=dict(color = 'rgb(255,255,255)'))
            
            fig2 = get_line(bad[op1].value_counts().reset_index())
            st.plotly_chart(go.Figure(data=fig1.data + fig2.data))

    with col2:
        st.text("Negative Review Graph")
        if op1 == "price":
            st.plotly_chart(get_line(bad[op1].value_counts().reset_index().sort_values(by='price', ascending=True)))
        else:
            st.plotly_chart(get_line(bad[op1].value_counts().reset_index()))

    
    # pie charts
    col1, col2 = st.columns(spec=2)

    with col1:
        st.text("Positive vs Negative")
        if op1 == "price":
            st.plotly_chart(get_pie(good[op1].value_counts().reset_index().sort_values(by='price', ascending=True)))
        else:
            st.plotly_chart(get_pie(good[op1].value_counts().reset_index()))

    with col2:
        st.text("Negative Graph")
        if op1 == "price":
            st.plotly_chart(get_pie(bad[op1].value_counts().reset_index().sort_values(by='price', ascending=True)))
        else:
            st.plotly_chart(get_pie(bad[op1].value_counts().reset_index()))

    
    if st.button("Get Insights"):
        st.code(f'''x_label = {op1} y_label = Count\n{get_insight(main, op1, 'count')}''')


    st.markdown("---")

elif radio == "Products":
    unit = st.selectbox("Enter sku", options=df['sku'].unique())

    g = df[df['sku'] == unit]
    col1, col2, col3, col4 = st.columns(spec=4)

    with col1:
        st.dataframe(g['states'].value_counts().reset_index(), width=300)
    with col2:
        # st.dataframe(g['states'].value_counts().reset_index())
        name, url, sku, price = np.array(df[df['sku'] == unit][['product_name', 'url','sku', 'price']].drop_duplicates()).reshape(-1)
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'lxml')
        img = soup.findAll('img')[5]['src']
        # st.text(img)
        st.image(img, caption=name+" "+str(price)+"rs", use_column_width=True)

    with col3:
        st.plotly_chart(get_bar(g['states'].value_counts().reset_index()))

    with col4:
        st.text(f"SKU = {sku}")
        st.text(f"Price = {price}rs")
        st.text(f"Total Sold Product Count = {g.shape[0]}")

    if senti[senti['sku'] == sku]['sentiment'].sum() <= 0: s = "Bad"
    else : s = "Good"
    st.text(f"Product Review by customer -> {s}")

    # st.dataframe(senti[senti['sku'] == sku]['sentiment'].reset_index())

    st.markdown("---")
    

else :
    # Options
    opt = [item.capitalize() for item in list(pd.get_dummies(df[radio.lower()]).columns)]
    option = st.selectbox("Category", options=opt)

    temp = arr.copy()
    temp.remove(radio.lower())
    # st.text(temp)
    for item in temp:
        # currently using dataframe
        current_df = df[df[radio.lower()] == option.lower()]

        # top5 hero products
        hero = list(current_df['sku'].value_counts().reset_index().head()['sku'])

        g = current_df[item].value_counts().reset_index()
        main = g
        
        if item == 'price':
            g = current_df[item].value_counts().reset_index().sort_values(by='price', ascending=True)
        

        col1, col2 = st.columns(spec=2)
        
        with col2:
            st.header("Top Products")

            value = st.selectbox("Select items by following : ", options=main[item])
            i = list(main[item]).index(value)

            sku_df = np.array(current_df[current_df[item] == main[item][i]]['sku'].value_counts().reset_index()['sku'])
            
            st.text(f"SKU's for {main[item][i]} (No of Products), {sku_df.shape[0]}")

            st.text("first few items")
            hero = sku_df.copy()[0:5]
            print_cols(hero)
            
            st.text("last few items")
            hero = sku_df.copy()[-5:]
            print_cols(hero)
        
        with col1:
            st.header(f"{item.capitalize()} wise")
            st.dataframe(main)
            g1 = current_df.copy()
            g2 = g1['sku'].value_counts().reset_index()
            g3 = g1.merge(right=g2).drop_duplicates(ignore_index=True)
            g3 = g3[g3[item] == value][['sku','product_name']].drop_duplicates(ignore_index=True)

            st.header(f"{item.capitalize()} wise")
            st.dataframe(g3)
            
        col1, col2 = st.columns(spec=2)
        with col1:
            st.plotly_chart(get_line(g))
        with col2:
            st.plotly_chart(get_pie(g))

        st.markdown("---")


