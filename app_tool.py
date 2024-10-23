import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
from langchain.llms import Ollama
import json

# 第一步：训练并保存回归模型
data_california = fetch_california_housing()
X_reg, y_reg = data_california.data, data_california.target
regressor = LinearRegression()
regressor.fit(X_reg, y_reg)
joblib.dump(regressor, 'regression_model.pkl')

# 第二步：训练并保存分类模型
data_iris = load_iris()
X_cls, y_cls = data_iris.data, data_iris.target
classifier = LogisticRegression(max_iter=200)
classifier.fit(X_cls, y_cls)
joblib.dump(classifier, 'classification_model.pkl')

# 第三步：加载模型
regression_model = joblib.load('regression_model.pkl')
classification_model = joblib.load('classification_model.pkl')

# 定义预测房价的工具函数
def predict_house_price(MedInc: float, HouseAge: float, AveRooms: float, AveBedrms: float,
                        Population: float, AveOccup: float, Latitude: float, Longitude: float) -> str:
    """预测房价。"""
    features = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    prediction = regression_model.predict([features])
    result = f"The predicted house price is: {prediction[0]:.2f} 10k USD"
    return result

# 定义预测鸢尾花种类的工具函数
def predict_iris_species(SepalLength: float, SepalWidth: float, PetalLength: float, PetalWidth: float) -> str:
    """预测鸢尾花的种类。"""
    features = [SepalLength, SepalWidth, PetalLength, PetalWidth]
    prediction = classification_model.predict([features])
    result = f"The predicted iris species is: {prediction[0]}"
    return result

# 一级页面：处理自然语言输入
def first_page():
    st.title("LLM Powered Prediction System")
    user_input = st.text_input("Enter your query (e.g., Predict house price or Predict iris species):")

    if st.button("Submit"):
        if user_input:
            llm = Ollama(model="llama3")
            prompt = f"""
You are an assistant with access to the following tools:
- predict_house_price: Predict house price, with parameters MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.
- predict_iris_species: Predict iris species, with parameters SepalLength, SepalWidth, PetalLength, PetalWidth.

Based on the user's input, return the tool name to use and the input. Your answer should be returned in JSON format containing the 'name' and 'arguments' keys. The value of 'arguments' should be a dictionary. Do not return any other text data.

User input: {user_input}
"""
            # 调用 LLM 来解析用户输入
            response = llm(prompt)

            try:
                tool_call = json.loads(response)
                tool_name = tool_call.get('name')
                arguments = tool_call.get('arguments')

                # 跳转到相应的预测页面
                if tool_name == "predict_house_price":
                    st.session_state.page = 'house_price'
                    st.session_state.arguments = arguments
                elif tool_name == "predict_iris_species":
                    st.session_state.page = 'iris_species'
                    st.session_state.arguments = arguments
                else:
                    st.write(f"No valid tool found for: {tool_name}")
            except json.JSONDecodeError:
                st.write("Failed to parse LLM response as JSON.")
                st.write(f"LLM response: {response}")

# 二级页面：房价预测页面
def house_price_page():
    st.title("House Price Prediction")

    # 使用 session_state 传递的参数填充表单
    arguments = st.session_state.get('arguments', {})
    MedInc = st.number_input("MedInc (Median Income):", value=arguments.get('MedInc', 0.0))
    HouseAge = st.number_input("HouseAge:", value=arguments.get('HouseAge', 0.0))
    AveRooms = st.number_input("AveRooms:", value=arguments.get('AveRooms', 0.0))
    AveBedrms = st.number_input("AveBedrms:", value=arguments.get('AveBedrms', 0.0))
    Population = st.number_input("Population:", value=arguments.get('Population', 0.0))
    AveOccup = st.number_input("AveOccup:", value=arguments.get('AveOccup', 0.0))
    Latitude = st.number_input("Latitude:", value=arguments.get('Latitude', 0.0))
    Longitude = st.number_input("Longitude:", value=arguments.get('Longitude', 0.0))

    if st.button("Predict House Price"):
        result = predict_house_price(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
        st.write(result)

    if st.button("Go back"):
        st.session_state.page = 'first'

# 二级页面：鸢尾花种类预测页面
def iris_species_page():
    st.title("Iris Species Prediction")

    # 使用 session_state 传递的参数填充表单
    arguments = st.session_state.get('arguments', {})
    SepalLength = st.number_input("SepalLength:", value=arguments.get('SepalLength', 0.0))
    SepalWidth = st.number_input("SepalWidth:", value=arguments.get('SepalWidth', 0.0))
    PetalLength = st.number_input("PetalLength:", value=arguments.get('PetalLength', 0.0))
    PetalWidth = st.number_input("PetalWidth:", value=arguments.get('PetalWidth', 0.0))

    if st.button("Predict Iris Species"):
        result = predict_iris_species(SepalLength, SepalWidth, PetalLength, PetalWidth)
        st.write(result)

    if st.button("Go back"):
        st.session_state.page = 'first'

# 主函数：根据页面导航不同内容
def main():
    # 初始化 session_state 的页面变量
    if 'page' not in st.session_state:
        st.session_state.page = 'first'

    # 页面路由
    if st.session_state.page == 'first':
        first_page()
    elif st.session_state.page == 'house_price':
        house_price_page()
    elif st.session_state.page == 'iris_species':
        iris_species_page()

if __name__ == "__main__":
    main()
