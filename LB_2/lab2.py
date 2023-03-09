#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:31:52 2023

@author: ksumasalitina
"""

# Імпорт бібліоткек
import streamlit as st
import matplotlib.pyplot as plt
from pycaret.datasets import get_data

# Отримання даних
dataset = get_data('pokemon')

# Виведення опису даних та шматочка таблиці
st.write("Даний датасет характерізує персонажів гри: їх імʼя, здібність, категорію, здоровʼя, силу атаки, швидкість, покоління, легендарність. Датасет має 800 записів та 13 атрибутів.")
st.table(dataset.head(6))

# Інтерактивна форма для вибору ознаки за якою буде побудована діаграма
options = ['Type 1', 'Type 2', 'Generation', 'Legendary']
default_option = 0  
selected_option = st.radio('Select an option', options, default_option)

# Підрахунок обʼєктів за обраною характеристикою 
class_counts = dataset[selected_option].value_counts()

# Створення кругової діаграми
fig, ax = plt.subplots()
ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',  textprops={'fontsize': 5})
ax.set_title('Distribution by {0}'.format(selected_option))

# Відображення діаграми
st.pyplot(fig)