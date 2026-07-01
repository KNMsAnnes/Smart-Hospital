       import streamlit as st
       import pandas as pd
       import pickle
       with open("hospital_model.pkl","rb") as f:
           bundle=pickle.load(f)
       model=bundle["model"]
       scaler=bundle["scaler"]
       features=bundle["features"]
       cols_to_scale=bundle["cols_to_scale"]
       gender_map=bundle["gender_map"]
       temp_map=bundle["temp_map"]
       hr_map=bundle["hr_map"]
       dur_map=bundle["dur_map"]
       cc_map=bundle["cc_map"]
       dept_map_inv=bundle["dept_map_inv"]
       st.title("AI Hospital Recommendation")
       age=st.number_input("Age",1,120,30)
       gender=st.selectbox("Gender",["Female","Male"])
       st.subheader("Symptoms")
       fever=st.checkbox("Fever")
       cough=st.checkbox("Cough")
       headache=st.checkbox("Headache")
       chest_pain=st.checkbox("Chest Pain")
       stomach_pain=st.checkbox("Stomach Pain")
       shortness_breath=st.checkbox("Shortness of Breath")
       nausea_vomiting=st.checkbox("Nausea / Vomiting")
       dizziness=st.checkbox("Dizziness")
       skin_rash=st.checkbox("Skin Rash")
       st.subheader("Condition")
       temperature=st.selectbox("Temperature",list(temp_map.keys()))
       heart_rate=st.selectbox("Heart Rate",list(hr_map.keys()))
       duration=st.selectbox("Duration",list(dur_map.keys()))
       chief=st.selectbox("Chief Complaint",list(cc_map.keys()))
       st.subheader("Medical History")
       asthma=st.checkbox("Asthma")
       hypertension=st.checkbox("Hypertension")
       heart_disease=st.checkbox("Heart Disease")
       if st.button("Predict"):
           patient=pd.DataFrame([{
               "age":age,
               "gender":gender_map[gender],
               "fever":int(fever),
               "cough":int(cough),
               "headache":int(headache),
               "chest_pain":int(chest_pain),
               "stomach_pain":int(stomach_pain),
               "shortness_breath":int(shortness_breath),
               "nausea_vomiting":int(nausea_vomiting),
               "dizziness":int(dizziness),
               "skin_rash":int(skin_rash),
               "temperature_level":temp_map[temperature],
               "heart_rate_level":hr_map[heart_rate],
               "duration":dur_map[duration],
               "asthma":int(asthma),
               "hypertension":int(hypertension),
               "heart_disease":int(heart_disease),
               "chief_complaint":cc_map[chief]
           }])
           scaled=patient.copy()
           scaled[cols_to_scale]=scaler.transform(patient[cols_to_scale])
           pred=model.predict(scaled[features])[0]
           conf=model.predict_proba(scaled[features])[0].max()*100
           st.success(f"Department: {dept_map_inv[pred]}")
           st.write(f"Confidence: {conf:.1f}%")
