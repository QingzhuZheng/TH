import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import matplotlib.pyplot as plt

# 自定义 CSS 样式
custom_css = """
<style>
    /* 设置标题颜色和样式 */
    h1 {
        color: #007BFF;
        text-align: center;
        font-size: 28px; /* 减小标题字体大小 */
    }
    /* 设置子标题颜色 */
    h2 {
        color: #6C757D;
    }
    /* 设置输入框和选择框样式 */
   .stTextInput>div>div>input,.stSelectbox>div>div>select {
        border-radius: 5px;
        border: 1px solid #CED4DA;
        padding: 5px;
    }
    /* 设置按钮样式 */
   .stButton>button {
        background-color: #28A745;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    /* 设置成功消息样式 */
   .stSuccess {
        background-color: #D4EDDA;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
    }
    /* 设置错误消息样式 */
   .stError {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 10px;
        border-radius: 5px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 加载数据
try:
    train_data = pd.read_csv("1.csv")
    test_data = pd.read_csv("2.csv")
    print("训练数据基本信息：")
    train_data.info()
    print("训练数据前几行：")
    print(train_data.head().to_csv(sep='\t', na_rep='nan'))

    print("测试数据基本信息：")
    test_data.info()
    print("测试数据前几行：")
    print(test_data.head().to_csv(sep='\t', na_rep='nan'))

except FileNotFoundError:
    st.error("未找到数据文件，请检查文件路径。")
    st.stop()

# 定义 LabelEncoder 对象
le_comb = LabelEncoder()
le_edema = LabelEncoder()
le_status = LabelEncoder()

# 对训练集进行编码
train_data['Comb'] = le_comb.fit_transform(train_data['Comb'])
train_data['Edema'] = le_edema.fit_transform(train_data['Edema'])
train_data['Status'] = le_status.fit_transform(train_data['Status'])

# 对测试集进行编码（使用训练集的编码器）
test_data['Comb'] = le_comb.transform(test_data['Comb'])
test_data['Edema'] = le_edema.transform(test_data['Edema'])
test_data['Status'] = le_status.transform(test_data['Status'])

# 确保特征和目标列正确
X_train = train_data.drop(columns=["Status"])
Y_train = train_data["Status"]
X_test = test_data.drop(columns=["Status"])
Y_test = test_data["Status"]

# 检查特征数
expected_features = 4
if X_train.shape[1] != expected_features or X_test.shape[1] != expected_features:
    st.error(f"特征数不符合预期，预期特征数为 {expected_features}。")
    st.stop()
print("X_train 特征数:", X_train.shape[1])
print("X_test 特征数:", X_test.shape[1])

random_seed = 316

# 使用最佳参数重新训练模型
rf_best = RandomForestClassifier(
    n_estimators=1000,
    max_features=2,
    min_samples_leaf=1,
    max_leaf_nodes=11,
    bootstrap=True,
    max_samples=66,
    random_state=random_seed
)

rf_best.fit(X_train, Y_train)

# Streamlit 应用
st.title("Random Forest model for predicting disease progression")

# 布局调整，输入在左侧，输出在右侧
col1, col2 = st.columns([1, 1])

# 输入特征（左侧列）
with col1:
    st.subheader("Please enter the MRE features in Random Forest model")
    input_features = {}
    for feature in X_train.columns:
        # 创建特征显示名称的映射
        display_name = feature
        if feature == 'ADC':
            display_name = 'Mural ADC'
        elif feature == 'Comb':
            display_name = 'Comb sign'
        elif feature == 'Edema':
            display_name = 'Mural edema'
        
        if feature == 'Comb':
            unique_values = le_comb.classes_
            input_features[feature] = le_comb.transform([st.selectbox(f"{display_name}", unique_values)])[0]
        elif feature == 'Edema':
            unique_values = le_edema.classes_
            input_features[feature] = le_edema.transform([st.selectbox(f"{display_name}", unique_values)])[0]
        else:
            input_features[feature] = st.number_input(f"{display_name}", value=0.0)

# 预测按钮（左侧列）
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_features])
        
        # 模型预测
        prediction = rf_best.predict(input_df)
        decoded_prediction = le_status.inverse_transform(prediction)[0]
        
        # 初始化 SHAP 解释器
        try:
            # 尝试使用最新 API
            explainer = shap.Explainer(rf_best.predict_proba, X_train)
            shap_values = explainer(input_df)
        except:
            # 回退到旧版 API
            st.warning("使用旧版 SHAP API 以兼容当前环境")
            explainer = shap.TreeExplainer(rf_best)
            shap_values = explainer.shap_values(input_df)
        
        # 获取预测类别的索引
        pred_class_index = prediction[0]
        
        # 显示结果（右侧列）
        with col2:
            st.success(f"Prediction: {decoded_prediction}")
            
            # 显示 waterfall 图（保留）
            st.subheader("SHAP summary plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            if isinstance(shap_values, list):
                # 旧版 API 返回列表
                shap.plots.waterfall(shap.Explainer(rf_best).shap_values(input_df)[pred_class_index][0])
            else:
                # 新版 API 返回 Explanation 对象
                shap.plots.waterfall(shap_values[0][:, pred_class_index])
            st.pyplot(fig)
            
            # 移除了 SHAP 特征重要性汇总图
            
    except Exception as e:
        with col2:
            st.error(f"预测过程中出现错误: {e}")
            