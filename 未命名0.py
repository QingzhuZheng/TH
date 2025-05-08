import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# 自定义 CSS 样式
custom_css = """
<style>
    /* 设置标题颜色和样式 */
    h1 {
        color: #007BFF;
        text-align: center;
    }
    /* 设置子标题颜色 */
    h2 {
        color: #6C757D;
    }
    /* 设置输入框和选择框样式 */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
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
    # 修改为相对路径，假设数据文件和脚本在同一目录下
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
    n_estimators=1000,  # 树的数量
    max_features=2,     # 每次分裂时随机选择的特征数
    min_samples_leaf=1,  # 每个叶子节点的最小样本数
    max_leaf_nodes=11,  # 每棵树的最大叶节点数
    bootstrap=True,     # 是否使用自助抽样
    max_samples=66,     # 每个树的训练样本数量（对应 R 中的 sampsize）
    random_state=random_seed
)

rf_best.fit(X_train, Y_train)

# Streamlit 应用
st.title("随机森林分类模型预测")

# 输入特征
st.subheader("请输入特征值")

# 使用两列布局
col1, col2 = st.columns(2)
with col1:
    for i, feature in enumerate(X_train.columns[:len(X_train.columns) // 2]):
        if feature == 'Comb':
            unique_values = le_comb.classes_
            input_features[feature] = le_comb.transform([st.selectbox(f"{feature}", unique_values)])[0]
        elif feature == 'Edema':
            unique_values = le_edema.classes_
            input_features[feature] = le_edema.transform([st.selectbox(f"{feature}", unique_values)])[0]
        else:
            input_features[feature] = st.number_input(f"{feature}", value=0.0)
with col2:
    for feature in X_train.columns[len(X_train.columns) // 2:]:
        if feature == 'Comb':
            unique_values = le_comb.classes_
            input_features[feature] = le_comb.transform([st.selectbox(f"{feature}", unique_values)])[0]
        elif feature == 'Edema':
            unique_values = le_edema.classes_
            input_features[feature] = le_edema.transform([st.selectbox(f"{feature}", unique_values)])[0]
        else:
            input_features[feature] = st.number_input(f"{feature}", value=0.0)

# 预测按钮
if st.button("进行预测"):
    try:
        input_df = pd.DataFrame([input_features])
        prediction = rf_best.predict(input_df)
        decoded_prediction = le_status.inverse_transform(prediction)[0]
        st.success(f"预测结果: {decoded_prediction}")
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")
    

