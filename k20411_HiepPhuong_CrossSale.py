
import streamlit as st
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title('DỰ ĐOÁN BÁN CHÉO')
st.write('''
# Dự đoán bán chéo từ các mô hình do nhóm thực hiện
---
''')

age = st.slider('Tuổi', 18, 100, 30)
vehicle_age = st.slider('Tuổi xe', 0.0, 5.0, 1.0, 0.5)
vehicle_damage = st.selectbox('Xe từng bị hư hỏng', ('Yes', 'No'))
previous_insurance = st.selectbox('Có mua bảo hiểm trước đó', ('Yes', 'No'))
button = st.button("Dự đoán")

st.write('''
# Kết quả
''')

side = st.sidebar
side.title('Nhóm Phạm Nguyễn Hiền Phương')
side.write('''
## Lớp K20411
''')
side.image('https://scontent.fsgn2-4.fna.fbcdn.net/v/t39.30808-6/315169072_502486661897201_7974025752979732935_n.jpg?_nc_cat=109&ccb=1-7&_nc_sid=730e14&_nc_ohc=lt9d3JHS99sAX_SZpC5&_nc_ht=scontent.fsgn2-4.fna&oh=00_AfCHnrQ1o-NQjS5Xvc0m_i0lBqYZQYkseKTK642mRaju4w&oe=63A70FFF')

result = {
    0: 'Không thể bán chéo',
    1: 'Có thể bán chéo'
}

@st.cache(allow_output_mutation=True)
def get_model():
    rfModel = joblib.load('model_Random_Forest_K20411.pkl')
    return rfModel

rfModel = get_model()

def predict(model, input_df):
    predictions = model.predict(input_df)
    return predictions[0]

def user_input_features():
    global vehicle_damage
    global previous_insurance
    vehicle_age_0, vehicle_age_1, vehicle_age_2 = [0.0, 0.0, 0.0]
    if vehicle_age < 1.0:
        vehicle_age_0 = 1.0
    elif vehicle_age < 2.0:
        vehicle_age_1 = 1.0
    else:
        vehicle_age_2 = 1.0
    previously_insured = 1.0 if previous_insurance == 'Yes' else 0.0
    vehicle_damage = 1.0 if vehicle_damage == 'Yes' else 0.0
    data = {'Age': age,
            'Vehicle_Age_0': vehicle_age_0,
            'Vehicle_Age_1': vehicle_age_1,
            'Vehicle_Age_2': vehicle_age_2,
            'Previously_Insured': previously_insured,
            'Vehicle_Damage': vehicle_damage}
    features = pd.DataFrame(data, index=[0])
    return features

def main():
    if button:
        input_df = user_input_features()
        prediction = predict(rfModel, input_df)
        st.write(result[prediction])

if __name__ == '__main__':
    main()
