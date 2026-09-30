import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json

from tensorflow import keras

st.set_page_config(
    page_title="Прогнозування діабету",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Прогнозування діабету")
st.write(
    "Веб-застосунок для прогнозування результату "
    "за допомогою багатошарової нейронної мережі." \
    "Для прогнозування використовується багатошарова нейронна мережа," \
    " навчена на наборі даних про діабет. Модель отримує 8 числових " \
    "характеристик та визначає один із двох класів: 0 або 1."
)

st.info(
    "Введіть параметри та натисніть кнопку «Зробити прогноз»."
)

# 5

@st.cache_resource
def load_model():
    return keras.models.load_model("diabetes_model.keras")


@st.cache_resource
def load_scaler():
    return joblib.load("diabetes_scaler.pkl")

# 11
@st.cache_data
def load_history():
    with open("history.json", "r") as f:
        return json.load(f)


model = load_model()
scaler = load_scaler()


# 6

st.header("Вхідні параметри")

col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Кількість вагітностей",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Рівень глюкози",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Артеріальний тиск",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Товщина шкіри",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )


with col2:

    insulin = st.number_input(
        "Інсулін",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "ІМТ (індекс маси тіла)",
        min_value=0.0,
        max_value=70.0,
        value=28.5,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Показник спадковості діабету",
        min_value=0.0,
        max_value=3.0,
        value=0.35,
        step=0.01
    )

    age = st.number_input(
        "Вік",
        min_value=1,
        max_value=120,
        value=35,
        step=1
    )


# 8
st.divider()


if st.button(
    "Зробити прогноз",
    use_container_width=True
):

    # 7
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })


    input_scaled = scaler.transform(input_data)


    prediction_probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]


    prediction = 1 if prediction_probability >= 0.5 else 0


    # 8

    st.subheader("Результат прогнозування")

    st.metric(
        "Ймовірність класу 1",
        f"{prediction_probability:.2%}"
    )

    if prediction == 1:

        st.warning(
            "Модель визначила результат: КЛАС 1"
        )

    else:

        st.success(
            "Модель визначила результат: КЛАС 0"
        )

    st.write(
        f"Отриманий результат класифікації: **{prediction}**"
    )


    # 9

    st.progress(
        float(prediction_probability)
    )

st.divider()

st.caption(
    "Модель розроблена та навчена в рамках лабораторної роботи №2. "
    "Веб-інтерфейс реалізовано за допомогою Streamlit."
)

# 9
history_data = load_history()

st.subheader("Динаміка навчання моделі")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Графік точності
ax1.plot(history_data["accuracy"], label="Навчальна вибірка")
ax1.plot(history_data["val_accuracy"], label="Валідаційна вибірка")
ax1.set_title("Зміна точності")
ax1.set_xlabel("Епоха")
ax1.set_ylabel("Точність")
ax1.legend()
ax1.grid(True)

# Графік втрат
ax2.plot(history_data["loss"], label="Навчальна вибірка")
ax2.plot(history_data["val_loss"], label="Валідаційна вибірка")
ax2.set_title("Зміна функції втрат")
ax2.set_xlabel("Епоха")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
st.pyplot(fig)