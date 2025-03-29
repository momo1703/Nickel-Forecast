from train_model_streamlit import train_lstm_model

if st.button("🔁 Retrain Model"):
    model = train_lstm_model(df)
