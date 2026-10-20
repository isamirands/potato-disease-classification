import streamlit as st
from campis.inference import Predictor
st.title("CAMPIS")
path=st.sidebar.text_input("Checkpoint", "artifacts/checkpoints/best_model.pt")
if __import__("os").path.exists(path):
    predictor=Predictor(path); crop=st.selectbox("Cultivo",list(predictor.crops),format_func=lambda x:predictor.crops[x]["display_name"]); image=st.file_uploader("Fotografía de la hoja",type=["jpg","jpeg","png","webp"])
    if image: st.image(image)
    if image and st.button("Analizar hoja"): st.json(predictor.predict(image,crop))
else: st.info("Entrene o indique un checkpoint para habilitar la demostración.")
st.caption("Predicción visual, no diagnóstico fitopatológico definitivo.")
