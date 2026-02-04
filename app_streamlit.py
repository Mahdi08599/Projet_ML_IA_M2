import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
Cette application utilise un modèle **Random Forest** optimisé (SMOTE + RobustScaler) 
pour détecter les transactions frauduleuses en temps réel.
""")

# =============================
# LOAD & TRAIN MODEL (Optimized)
# =============================
@st.cache_resource 
def load_and_train():
    df = pd.read_csv("creditcard.csv")
    
    # 1. Feature Engineering (Comme à l'entraînement)
    df['log_Amount'] = np.log1p(df['Amount'])
    X = df.drop(["Class", "Time", "Amount"], axis=1) # On garde 29 colonnes (V1-V28 + log_Amount)
    y = df["Class"]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 4. SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    # 5. Random Forest (Tes hyperparamètres optimisés)
    model = RandomForestClassifier(
        n_estimators=50,    
        max_depth=20,       # Évite l'overfitting
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_sm, y_train_sm)

    return model, scaler, df # On retourne aussi le df original pour les exemples

with st.spinner('Entraînement du modèle en cours... (ça peut prendre 30s)'):
    model, scaler, df_orig = load_and_train()

st.success("Modèle chargé avec succès !")

# =============================
# INTERFACE
# =============================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simuler une transaction")
    
    # Boutons pour charger des exemples réels
    st.write("Charger un profil type :")
    b_col1, b_col2 = st.columns(2)
    
    if b_col1.button("👤 Client Normal"):
        # On prend une ligne aléatoire de classe 0
        example = df_orig[df_orig.Class == 0].sample(1).iloc[0]
        st.session_state["inputs"] = example
    
    if b_col2.button("⚠️ Fraudeur"):
        # On prend une ligne aléatoire de classe 1
        example = df_orig[df_orig.Class == 1].sample(1).iloc[0]
        st.session_state["inputs"] = example

    # Récupération des inputs (V1 à V28 + Amount)
    # On gère l'Amount séparément pour l'UX
    input_data = {}
    
    # Si on a chargé un exemple, on l'utilise, sinon valeurs par défaut
    defaults = st.session_state.get("inputs", None)
    
    amount_val = float(defaults['Amount']) if defaults is not None else 0.0
    input_amount = st.number_input("Montant de la transaction ($)", value=amount_val)
    
    with st.expander("Voir les paramètres techniques (V1-V28)"):
        for i in range(1, 29):
            col_name = f"V{i}"
            val = float(defaults[col_name]) if defaults is not None else 0.0
            input_data[col_name] = st.number_input(f"{col_name}", value=val, key=col_name)

with col2:
    st.subheader("Analyse en Temps Réel")
    
    if st.button("Lancer la détection", type="primary"):
        # 1. Reconstruire le vecteur de features comme à l'entraînement
        # On a besoin de [V1, ..., V28, log_Amount]
        
        features = list(input_data.values()) # V1 à V28
        features.append(np.log1p(input_data.get('Amount', input_amount))) # Ajout du log_Amount
        
        # Conversion en array et scaling
        X_new = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X_new)

        # Prédiction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        # Affichage
        st.divider()
        if probability > 0.5: # Tu peux ajuster ce seuil (ex: 0.4)
            st.error(f"🚨 ALERTE FRAUDE DÉTECTÉE")
            st.metric(label="Probabilité de fraude", value=f"{probability:.1%}")
            st.write("Le modèle recommande de bloquer cette transaction.")
        else:
            st.success(f"✅ Transaction Légitime")
            st.metric(label="Risque estimé", value=f"{probability:.1%}")
            st.write("Aucune anomalie détectée.")