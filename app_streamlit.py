import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm # Important pour la couleur de la matrice

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Fraud Detection Demo", layout="wide", page_icon="💳")
st.title("💳 Détection de Fraude par Carte Bancaire")
st.markdown("""
*Projet de fin d'études (Master 2)*. 
Cette application démontre un modèle **Random Forest** entraîné sur des données déséquilibrées 
(utilisant SMOTE et RobustScaler) et validé rigoureusement sur un jeu de test indépendant.
""")

# =============================
# LOAD & TRAIN MODEL (Rigoureux)
# =============================
@st.cache_resource 
def load_and_train():
    # Chargement
    try:
        df = pd.read_csv("creditcard.csv")
    except FileNotFoundError:
        st.error("Fichier 'creditcard.csv' introuvable. Veuillez le charger sur GitHub.")
        st.stop()
    
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

    return model, scaler, df, X_test, y_test

with st.spinner('Initialisation du modèle et validation scientifique...'):
    model, scaler, df_orig, X_test, y_test = load_and_train()

st.success("Modèle prêt et validé sur les données de test !")

# =============================
# INTERFACE DE SIMULATION
# =============================
st.divider()
st.header("🕵️‍♀️ Simulateur de Transaction")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Paramètres de la transaction")
    st.write("Profils rapides :")
    b_col1, b_col2 = st.columns(2)
    
    if b_col1.button("👤 Normal"):
        st.session_state["inputs"] = df_orig[df_orig.Class == 0].sample(1).iloc[0]
    
    if b_col2.button("⚠️ Fraudeur"):
        st.session_state["inputs"] = df_orig[df_orig.Class == 1].sample(1).iloc[0]

    defaults = st.session_state.get("inputs", None)
    
    amount_val = float(defaults['Amount']) if defaults is not None else 0.0
    input_amount = st.number_input("Montant de la transaction (€/$)", value=amount_val, step=10.0)
    
    input_data = {}
    with st.expander("Afficher les variables techniques anonymisées (V1-V28)"):
        for i in range(1, 29):
            col_name = f"V{i}"
            val = float(defaults[col_name]) if defaults is not None else 0.0
            input_data[col_name] = st.number_input(f"{col_name}", value=val, key=col_name, format="%.4f")

with col2:
    st.subheader("Résultat de l'analyse")
    st.write("Cliquez ci-dessous pour soumettre la transaction au modèle.")
    if st.button("Lancer la détection", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            # Préparation des features
            features = [input_data[f"V{i}"] for i in range(1, 29)]
            features.append(np.log1p(input_amount))
            
            X_new = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X_new)
            probability = model.predict_proba(X_scaled)[0][1]

            st.divider()
            if probability > 0.5:
                st.error(f"🚨 **ALERTE FRAUDE** (Probabilité: {probability:.1%})")
                st.markdown("### Recommandation : **BLOQUER IMMÉDIATEMENT**")
                st.write("Le modèle a détecté des motifs caractéristiques d'une transaction frauduleuse.")
            else:
                st.success(f"✅ **Transaction Légitime** (Risque: {probability:.1%})")
                st.markdown("### Recommandation : **AUTORISER**")
                st.write("Aucune anomalie significative détectée.")

# =============================
# PERFORMANCE REELLE (Test Set)
# =============================
st.divider()
st.header("📊 Validation Scientifique (Données de Test - 20%)")
st.markdown("Ces résultats sont calculés sur **56 962 transactions jamais vues** par le modèle pendant l'entraînement.")

# Calcul des prédictions sur le VRAI jeu de test
X_test_scaled = scaler.transform(X_test)
y_pred_test = model.predict(X_test_scaled)

tab1, tab2, tab3, tab4 = st.tabs(["Matrice de Confusion", "Métriques Clés", "Feature Importance", "Distribution Montants"])

with tab1:
    st.subheader("Performance sur les données de test")
    
    cm = confusion_matrix(y_test, y_pred_test)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 1. Fond coloré avec échelle Logarithmique pour voir les nuances
    sns.heatmap(cm, annot=False, cmap='Blues', ax=ax, cbar=False, norm=LogNorm())
    
    # 2. Texte forcé en NOIR GRAS par-dessus
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                    ha='center', va='center',
                    color='black', fontsize=18, fontweight='bold')
    
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Normal', 'Fraude'])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Normal', 'Fraude'])
    ax.set_xlabel('Prédiction du Modèle')
    ax.set_ylabel('Réalité (Données)')
    st.pyplot(fig)
    
    st.markdown(f"""
    **Interprétation pour le Jury :**
    * Le dataset de test contient **{cm[1][0] + cm[1][1]} fraudes réelles**.
    * Le modèle en a détecté **{cm[1][1]}** (Vrais Positifs).
    * Il en a manqué **{cm[1][0]}** (Faux Négatifs).
    * Malgré le fort déséquilibre, le modèle maintient un taux de fausses alertes très bas (**{cm[0][1]}** sur +56k transactions).
    """)

with tab2:
    st.subheader("Indicateurs de Performance")
    
    rec = recall_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Rappel (Recall)", f"{rec:.1%}", help="Pourcentage des fraudes réelles détectées (Le plus important).")
    col_m2.metric("Précision", f"{prec:.1%}", help="Quand le modèle dit 'Fraude', a-t-il raison ?")
    col_m3.metric("F1-Score", f"{f1:.1%}", help="Moyenne harmonique Précision/Rappel.")
    
    st.info("Ces métriques confirment la capacité du modèle à généraliser sur de nouvelles données.")

with tab3:
    st.subheader("Qu'est-ce qui définit une fraude ?")
    importances = model.feature_importances_
    feature_names = [f"V{i}" for i in range(1, 29)] + ["log_Amount"]
    
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
    
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax3)
    ax3.set_title("Top 10 des variables les plus prédictives")
    st.pyplot(fig3)
    st.caption("Ces variables (issues de la PCA) sont celles que le modèle utilise le plus pour prendre sa décision.")

with tab4:
    st.subheader("Pourquoi utiliser le Logarithme sur le Montant ?")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.kdeplot(df_orig[df_orig.Class == 0]['log_Amount'], label='Transaction Normale', fill=True, color="tab:blue", ax=ax2)
    sns.kdeplot(df_orig[df_orig.Class == 1]['log_Amount'], label='Fraude', fill=True, color="tab:orange", ax=ax2)
    ax2.set_title("Distribution des montants (Échelle Logarithmique)")
    ax2.set_xlabel("log(Montant + 1)")
    ax2.legend()
    st.pyplot(fig2)
    st.caption("On observe que les fraudes (orange) ne suivent pas la même distribution de montant que les transactions normales (bleu), ce qui aide le modèle.")