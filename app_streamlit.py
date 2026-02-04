import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from matplotlib.colors import LogNorm  

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
Cette application permet de visualiser la performance du modèle sur des données de test (20% du dataset) 
et de simuler des transactions.
""")

# =============================
# LOAD & TRAIN MODEL (Rigoureux)
# =============================
@st.cache_resource 
def load_and_train():
    # Chargement
    df = pd.read_csv("creditcard.csv")
    
    # 1. Feature Engineering
    df['log_Amount'] = np.log1p(df['Amount'])
    X = df.drop(["Class", "Time", "Amount"], axis=1)
    y = df["Class"]

    # 2. Split (Identique à train_model.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Important : On garde le X_test pour la validation visuelle plus tard
    
    # 4. SMOTE (Uniquement sur le train)
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    # 5. Random Forest
    model = RandomForestClassifier(
        n_estimators=50,    
        max_depth=20,       
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_sm, y_train_sm)

    # On retourne tout ce dont on a besoin : le modèle, le scaler, le df original (pour la démo)
    # ET surtout X_test/y_test pour la matrice de confusion
    return model, scaler, df, X_test, y_test

with st.spinner('Entraînement et validation du modèle en cours...'):
    model, scaler, df_orig, X_test, y_test = load_and_train()

st.success("Modèle entraîné et validé sur le jeu de test !")

# =============================
# INTERFACE DE SIMULATION
# =============================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simuler une transaction")
    st.write("Charger un profil type :")
    b_col1, b_col2 = st.columns(2)
    
    if b_col1.button("👤 Client Normal"):
        st.session_state["inputs"] = df_orig[df_orig.Class == 0].sample(1).iloc[0]
    
    if b_col2.button("⚠️ Fraudeur"):
        st.session_state["inputs"] = df_orig[df_orig.Class == 1].sample(1).iloc[0]

    defaults = st.session_state.get("inputs", None)
    
    amount_val = float(defaults['Amount']) if defaults is not None else 0.0
    input_amount = st.number_input("Montant ($)", value=amount_val)
    
    input_data = {}
    with st.expander("Voir les variables techniques"):
        for i in range(1, 29):
            col_name = f"V{i}"
            val = float(defaults[col_name]) if defaults is not None else 0.0
            input_data[col_name] = st.number_input(f"{col_name}", value=val, key=col_name)

with col2:
    st.subheader("Analyse en Temps Réel")
    if st.button("Lancer la détection", type="primary"):
        features = [input_data[f"V{i}"] for i in range(1, 29)]
        features.append(np.log1p(input_amount))
        
        X_new = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X_new)
        probability = model.predict_proba(X_scaled)[0][1]

        st.divider()
        if probability > 0.5:
            st.error(f"🚨 ALERTE FRAUDE (Probabilité: {probability:.1%})")
            st.write("Transaction suspecte bloquée.")
        else:
            st.success(f"✅ Transaction Légitime (Risque: {probability:.1%})")

# =============================
# PERFORMANCE REELLE (Test Set)
# =============================
st.divider()
st.header("📊 Validation Scientifique (Sur X_test)")

# Prédictions sur le jeu de test
X_test_scaled = scaler.transform(X_test)
y_pred_test = model.predict(X_test_scaled)
y_proba_test = model.predict_proba(X_test_scaled)[:, 1]

tab1, tab2, tab3 = st.tabs(["Matrice de Confusion", "Métriques Clés", "Feature Importance"])

with tab1:
    st.subheader("Performance sur les données jamais vues (20% du dataset)")
    
    cm = confusion_matrix(y_test, y_pred_test)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                norm=LogNorm(),  # <--- Magie : permet de voir les nuances même entre 56000 et 84
                xticklabels=['Normal', 'Fraude'], 
                yticklabels=['Normal', 'Fraude'],
                annot_kws={"size": 16, "weight": "bold"}, # On laisse Seaborn gérer la couleur (Noir/Blanc)
                cbar=False)
    
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Réalité')
    st.pyplot(fig)
    
    st.markdown(f"""
    **Analyse Scientifique :**
    Le modèle a été testé sur **{len(y_test)}** transactions réelles (20% du dataset).
    * Sur **{cm[1][0] + cm[1][1]}** cas de fraudes réelles :
        * ✅ **{cm[1][1]}** ont été stoppées (Vrais Positifs).
        * ⚠️ **{cm[1][0]}** sont passées au travers (Faux Négatifs).
    * **Conclusion :** Avec un taux de détection d'environ **{cm[1][1]/(cm[1][0]+cm[1][1]):.1%}**, le modèle est performant pour sécuriser les transactions.
    """)

with tab2:
    st.subheader("Indicateurs de Performance")
    
    col_met1, col_met2, col_met3 = st.columns(3)
    
    # Calcul des métriques
    rec = recall_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    col_met1.metric("Rappel (Recall)", f"{rec:.1%}", help="Capacité à trouver les fraudes")
    col_met2.metric("Précision", f"{prec:.1%}", help="Fiabilité des alertes")
    col_met3.metric("F1-Score", f"{f1:.1%}", help="Équilibre Précision/Rappel")
    
    st.info("Ces résultats correspondent exactement à ceux obtenus dans le script d'entraînement.")

with tab3:
    st.subheader("Variables Déterminantes")
    importances = model.feature_importances_
    feature_names = [f"V{i}" for i in range(1, 29)] + ["log_Amount"]
    
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
    
    fig3, ax3 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax3)
    st.pyplot(fig3)