import streamlit as st
import pickle
import pandas as pd

# Load model
with open('best_model.pkl', 'rb') as f:
    saved = pickle.load(f)

model = saved['model']
user_encoder = saved['user_encoder']
recipe_encoder = saved['recipe_encoder']
df = saved['df']

st.title("🍽️ Food Recipe Recommendation System")

user_input = st.text_input("U001:")

if user_input:
    try:
        user_id_enc = user_encoder.transform([user_input])[0]
    except:
        st.error("Invalid User ID.")
    else:
        recipe_ids = df['recipe_enc'].unique()
        preds = []
        for rid in recipe_ids:
            est = model.predict(user_id_enc, rid).est
            preds.append((rid, est))
        
        top5 = sorted(preds, key=lambda x: x[1], reverse=True)[:5]

        st.subheader("Top 5 Recommended Recipes:")
        for rid, score in top5:
            recipe_id = recipe_encoder.inverse_transform([rid])[0]
            info = df[df['recipe_id'] == recipe_id].iloc[0]
            st.markdown(f"**🍛 Recipe:** {info['recipe_name']}")
            st.markdown(f"**📝 Ingredients:** {info['ingredients']}")
            st.markdown(f"**⭐ Predicted Rating:** {score:.2f}")
            st.markdown("---")
