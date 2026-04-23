import os
import io
import base64
import warnings
import logging
import time
import shutil
import platform
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

import plotly.express as px
import plotly.graph_objects as go

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

warnings.filterwarnings('ignore')

# ---------- CONFIGURATION DES LOGS ----------
logging.basicConfig(
    filename='agristat.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def log_action(action, details=""):
    logging.info(f"{action} - {details}")
    print(f"📝 {action}")

# ---------- INITIALISATION FLASK ----------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'agristat_secret_key_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agristat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- SYSTÈME DE CACHE ----------
_plot_cache = {}
_cache_time = {}
CACHE_DURATION = 300

def get_cached_plot(key, generate_func):
    current_time = time.time()
    if key in _plot_cache and (current_time - _cache_time.get(key, 0) < CACHE_DURATION):
        return _plot_cache[key]
    plot = generate_func()
    _plot_cache[key] = plot
    _cache_time[key] = current_time
    return plot

def clear_cache():
    global _plot_cache, _cache_time
    _plot_cache.clear()
    _cache_time.clear()

# ---------- CALENDRIER CULTURAL ----------
CULTURAL_CALENDAR = {
    'Maïs': {'semis': 'Mars-Avril', 'recolte': 'Juillet-Août', 'cycle': '90-120 jours'},
    'Cacao': {'semis': 'Avril-Juin', 'recolte': 'Octobre-Mars', 'cycle': '5-6 ans'},
    'Banane Plantain': {'semis': 'Toute l\'année', 'recolte': '8-10 mois après', 'cycle': '8-10 mois'},
    'Manioc': {'semis': 'Mars-Mai', 'recolte': '12-18 mois après', 'cycle': '12-18 mois'},
    'Tomate': {'semis': 'Février-Mars', 'recolte': 'Mai-Juillet', 'cycle': '90-120 jours'},
    'Arachide': {'semis': 'Avril-Mai', 'recolte': 'Août-Septembre', 'cycle': '120-150 jours'},
    'Coton': {'semis': 'Mai-Juin', 'recolte': 'Novembre-Décembre', 'cycle': '150-180 jours'}
}

# ---------- MODÈLE DE DONNÉES ----------
class AgriData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), nullable=False, default=datetime.now().strftime("%Y-%m-%d"))
    region = db.Column(db.String(50), nullable=False, default="Centre")
    type_culture = db.Column(db.String(50), nullable=False)
    surface_ha = db.Column(db.Float, nullable=False)
    pluviometrie_mm = db.Column(db.Float, nullable=False)
    engrais_kg = db.Column(db.Float, nullable=False)
    rendement_tonnes = db.Column(db.Float, nullable=False)
    qualite = db.Column(db.String(20), nullable=True)
    prix_vente_fcfa = db.Column(db.Float, nullable=True, default=0)

# ---------- VALIDATION ----------
def validate_input(data):
    errors = []
    cultures_valides = ['Maïs', 'Cacao', 'Banane Plantain', 'Manioc', 'Tomate', 'Arachide', 'Coton']
    
    if data.get('type_culture') not in cultures_valides:
        errors.append("🌱 Culture non reconnue")
    
    surface = data.get('surface_ha', 0)
    if surface <= 0:
        errors.append("📐 La surface doit être positive")
    elif surface > 1000:
        errors.append("📐 Surface maximale : 1000 hectares")
    
    pluie = data.get('pluviometrie_mm', -1)
    if pluie < 0:
        errors.append("🌧️ La pluviométrie ne peut pas être négative")
    elif pluie > 3000:
        errors.append("🌧️ Pluviométrie maximale : 3000 mm")
    
    engrais = data.get('engrais_kg', -1)
    if engrais < 0:
        errors.append("🧪 L'engrais ne peut pas être négatif")
    elif engrais > 2000:
        errors.append("🧪 Engrais maximal : 2000 kg/ha")
    
    rendement = data.get('rendement_tonnes', -1)
    if rendement <= 0:
        errors.append("🌾 Le rendement doit être positif")
    elif rendement > 100:
        errors.append("🌾 Rendement maximal : 100 tonnes/ha")
    
    if surface > 0 and rendement > 0:
        production_totale = surface * rendement
        if production_totale > 10000:
            errors.append(f"⚠️ Production totale ({production_totale:.0f} tonnes) semble irréaliste")
    
    return errors

# ---------- RECOMMANDATIONS PERSONNALISÉES ----------
def get_personalized_recommendation(entry):
    recs = []
    
    if entry.pluviometrie_mm < 900:
        recs.append("💧 Pluviométrie très faible. Irrigation fortement recommandée.")
    elif entry.pluviometrie_mm < 1100:
        recs.append("💧 Pluviométrie modérée. Surveillez l'humidité du sol.")
    else:
        recs.append("✅ Pluviométrie satisfaisante.")
    
    if entry.engrais_kg < 150:
        recs.append("🧪 Engrais insuffisant. Augmentez à 200-300 kg/ha pour de meilleurs rendements.")
    elif entry.engrais_kg > 400:
        recs.append("⚠️ Engrais excessif. Risque de pollution et coûts inutiles.")
    else:
        recs.append("✅ Dosage d'engrais optimal.")
    
    if entry.prix_vente_fcfa and entry.prix_vente_fcfa > 0:
        revenu = entry.surface_ha * entry.rendement_tonnes * 1000 * entry.prix_vente_fcfa
        recs.append(f"💰 Revenu estimé : {revenu:,.0f} FCFA")
    
    return recs

# ---------- SAUVEGARDE ET EXPORT ----------
def backup_database():
    try:
        backup_dir = 'backups'
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{backup_dir}/agristat_backup_{timestamp}.db"
        shutil.copy('agristat.db', backup_file)
        return True, backup_file
    except:
        return False, None

def export_data_csv():
    try:
        data = AgriData.query.all()
        df = pd.DataFrame([{
            'date': d.date, 'region': d.region, 'culture': d.type_culture,
            'surface_ha': d.surface_ha, 'pluviometrie_mm': d.pluviometrie_mm,
            'engrais_kg': d.engrais_kg, 'rendement_tonnes': d.rendement_tonnes, 
            'qualite': d.qualite, 'prix_fcfa': d.prix_vente_fcfa or 0
        } for d in data])
        export_dir = 'exports'
        os.makedirs(export_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"{export_dir}/agristat_export_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        return csv_file
    except:
        return None

# ---------- DONNÉES DE DÉMO ----------
def generate_demo_data():
    np.random.seed(42)
    demo_data = []
    cultures = ['Maïs', 'Cacao', 'Banane Plantain', 'Manioc', 'Tomate']
    regions = ['Centre', 'Littoral', 'Ouest', 'Sud', 'Est']
    prix_ref = {'Maïs': 250, 'Cacao': 1500, 'Banane Plantain': 300, 'Manioc': 200, 'Tomate': 500}
    
    for i in range(60):
        culture = np.random.choice(cultures)
        region = np.random.choice(regions)
        surface = np.random.uniform(0.5, 20)
        pluie = np.random.uniform(800, 1800)
        engrais = np.random.uniform(50, 400)
        
        base = {'Maïs': 2.5, 'Cacao': 0.8, 'Banane Plantain': 15, 'Manioc': 12, 'Tomate': 20}.get(culture, 5)
        rendement = base + (pluie - 1000) * 0.002 + (engrais - 200) * 0.01 + np.random.normal(0, base * 0.1)
        rendement = max(base * 0.5, min(rendement, base * 1.8))
        
        qualite = 'Faible' if rendement < base * 0.7 else ('Moyenne' if rendement < base * 1.2 else 'Bonne')
        prix = prix_ref.get(culture, 300)
        
        demo_data.append(AgriData(
            region=region, type_culture=culture,
            surface_ha=round(surface, 2), pluviometrie_mm=round(pluie, 1),
            engrais_kg=round(engrais, 1), rendement_tonnes=round(rendement, 2), 
            qualite=qualite, prix_vente_fcfa=prix
        ))
    return demo_data

# ---------- CRÉATION BDD ----------
with app.app_context():
    try:
        db.create_all()
        count = AgriData.query.count()
        print(f"📊 Base de données initialisée. {count} parcelles existantes.")
    except:
        db.drop_all()
        db.create_all()
        count = 0
        print("📊 Nouvelle base créée.")
    
    if count == 0:
        print("🌾 Génération des données agricoles de démonstration...")
        demo = generate_demo_data()
        for d in demo:
            db.session.add(d)
        db.session.commit()
        print(f"✅ {len(demo)} parcelles ajoutées")

# ---------- FONCTION GRAPHIQUE ----------
def create_plot():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120, facecolor='white')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

# ---------- RECOMMANDATIONS GLOBALES ----------
def generate_recommendations(df):
    recommendations = []
    avg_pluie = df['Pluviométrie (mm)'].mean()
    avg_engrais = df['Engrais (kg/ha)'].mean()
    
    if avg_pluie < 1000:
        recommendations.append({'icon': '💧', 'title': 'Irrigation recommandée', 'desc': f'Pluviométrie moyenne ({avg_pluie:.0f} mm) inférieure au seuil optimal.', 'color': 'info'})
    else:
        recommendations.append({'icon': '✅', 'title': 'Pluviométrie satisfaisante', 'desc': f'Pluviométrie moyenne ({avg_pluie:.0f} mm) dans la norme.', 'color': 'success'})
    
    if avg_engrais < 200:
        recommendations.append({'icon': '🧪', 'title': 'Augmenter les intrants', 'desc': f'Utilisation d\'engrais ({avg_engrais:.0f} kg/ha) inférieure à la moyenne.', 'color': 'warning'})
    elif avg_engrais > 500:
        recommendations.append({'icon': '⚠️', 'title': 'Surplus d\'engrais', 'desc': f'Utilisation d\'engrais élevée. Risque de pollution.', 'color': 'danger'})
    else:
        recommendations.append({'icon': '✅', 'title': 'Gestion d\'engrais optimale', 'desc': f'Utilisation d\'engrais ({avg_engrais:.0f} kg/ha) dans la norme.', 'color': 'success'})
    
    best_culture = df.groupby('Culture')['Rendement (t/ha)'].mean().idxmax()
    best_rendement = df.groupby('Culture')['Rendement (t/ha)'].mean().max()
    recommendations.append({'icon': '🏆', 'title': f'Culture star : {best_culture}', 'desc': f'Rendement moyen de {best_rendement:.1f} t/ha.', 'color': 'success'})
    
    return recommendations

# ---------- ANALYSE COMPLÈTE ----------
def generate_full_analysis():
    data = AgriData.query.all()
    
    if len(data) < 10:
        return {'error': 'not_enough_data', 'count': len(data)}
    
    try:
        df_data = []
        for d in data:
            df_data.append({
                'Surface (ha)': d.surface_ha, 'Pluviométrie (mm)': d.pluviometrie_mm,
                'Engrais (kg/ha)': d.engrais_kg, 'Rendement (t/ha)': d.rendement_tonnes,
                'Qualité': d.qualite, 'Culture': d.type_culture, 'Région': d.region,
                'Date': d.date
            })
        df = pd.DataFrame(df_data)
        
        results = {}
        results['count'] = len(df)
        results['cultures'] = df['Culture'].unique().tolist()
        results['regions'] = df['Région'].unique().tolist()
        results['recommendations'] = generate_recommendations(df)
        results['desc'] = df[['Surface (ha)', 'Pluviométrie (mm)', 'Engrais (kg/ha)', 'Rendement (t/ha)']].describe().to_html(classes='table table-striped table-sm', border=0)
        
        features = ['Surface (ha)', 'Pluviométrie (mm)', 'Engrais (kg/ha)']
        X = df[features].values
        y = df['Rendement (t/ha)'].values
        
        # Module 1 : Régression simple
        try:
            X_simple = df[['Pluviométrie (mm)']].values
            model_simple = LinearRegression()
            model_simple.fit(X_simple, y)
            y_pred = model_simple.predict(X_simple)
            r2_simple = r2_score(y, y_pred)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(X_simple, y, alpha=0.6, c='#2E7D32', edgecolor='white', s=70)
            plt.plot(X_simple, y_pred, color='#FF8F00', linewidth=2, label=f'R²={r2_simple:.3f}')
            plt.xlabel('Pluviométrie (mm/mois)', fontsize=12)
            plt.ylabel('Rendement (tonnes/ha)', fontsize=12)
            plt.title('Module 1 : Régression Simple', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            results['reg_simple_plot'] = create_plot()
            results['reg_simple_eq'] = f"Rendement = {model_simple.intercept_:.2f} + {model_simple.coef_[0]:.4f} × Pluviométrie"
            results['reg_simple_r2'] = round(r2_simple, 3)
        except:
            results['reg_simple_plot'] = ''
            results['reg_simple_eq'] = 'Erreur'
            results['reg_simple_r2'] = 0
        
        # Module 2 : Régression multiple
        try:
            model_multi = LinearRegression()
            model_multi.fit(X, y)
            y_pred_multi = model_multi.predict(X)
            r2_multi = r2_score(y, y_pred_multi)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y, y_pred_multi, alpha=0.6, c='#1B5E20', edgecolor='white', s=70)
            min_v, max_v = min(y.min(), y_pred_multi.min()), max(y.max(), y_pred_multi.max())
            plt.plot([min_v, max_v], [min_v, max_v], '#FF8F00', linestyle='--', linewidth=2)
            plt.xlabel('Rendement réel (t/ha)', fontsize=12)
            plt.ylabel('Rendement prédit (t/ha)', fontsize=12)
            plt.title(f'Module 2 : Régression Multiple\nR²={r2_multi:.3f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            results['reg_multi_plot'] = create_plot()
            coef = model_multi.coef_
            results['reg_multi_eq'] = f"Rendement = {model_multi.intercept_:.2f} + {coef[0]:.3f}×Surface + {coef[1]:.4f}×Pluie + {coef[2]:.3f}×Engrais"
            results['reg_multi_r2'] = round(r2_multi, 3)
            results['model_coef'] = {'intercept': model_multi.intercept_, 'surface': coef[0], 'pluie': coef[1], 'engrais': coef[2]}
        except:
            results['reg_multi_plot'] = ''
            results['reg_multi_eq'] = 'Erreur'
            results['reg_multi_r2'] = 0
            results['model_coef'] = {'intercept': 2.5, 'surface': 0.1, 'pluie': 0.002, 'engrais': 0.01}
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features + ['Rendement (t/ha)']].values)
        
        # Module 3 : ACP
        try:
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            pca_df['Qualité'] = df['Qualité']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            colors = {'Faible': '#C62828', 'Moyenne': '#F9A825', 'Bonne': '#2E7D32'}
            for qualite in colors:
                idx = pca_df['Qualité'] == qualite
                if idx.any():
                    ax1.scatter(pca_df.loc[idx, 'PC1'], pca_df.loc[idx, 'PC2'], c=colors[qualite], label=qualite, alpha=0.7, edgecolor='white', s=70)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax1.set_title('Projection ACP des parcelles')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.bar(range(1, 3), pca.explained_variance_ratio_, color='#2E7D32')
            ax2.plot(range(1, 3), np.cumsum(pca.explained_variance_ratio_), 'ro-', label='Cumul')
            ax2.set_xlabel('Composante Principale')
            ax2.set_ylabel('Variance expliquée')
            ax2.set_title('Variance par composante')
            ax2.legend()
            ax2.set_xticks([1, 2])
            plt.tight_layout()
            results['pca_plot'] = create_plot()
            results['pca_var'] = round(sum(pca.explained_variance_ratio_[:2])*100, 1)
        except:
            results['pca_plot'] = ''
            results['pca_var'] = 0
        
        # Module 4 : Classification
        try:
            le = LabelEncoder()
            y_class = le.fit_transform(df['Qualité'])
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_class, test_size=0.3, random_state=42)
            
            logreg = LogisticRegression(multi_class='ovr', max_iter=1000)
            logreg.fit(X_train, y_train)
            acc_log = accuracy_score(y_test, logreg.predict(X_test))
            
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            acc_knn = accuracy_score(y_test, knn.predict(X_test))
            
            results['class_sup_log_acc'] = round(acc_log*100, 1)
            results['class_sup_knn_acc'] = round(acc_knn*100, 1)
            results['classes_labels'] = le.classes_.tolist()
        except:
            results['class_sup_log_acc'] = 0
            results['class_sup_knn_acc'] = 0
            results['classes_labels'] = []
        
        # Module 5 : Clustering
        try:
            inertias = []
            K_range = range(1, 8)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(K_range, inertias, 'o-', color='#2E7D32', markersize=8)
            ax1.set_xlabel('Nombre de clusters K')
            ax1.set_ylabel('Inertie')
            ax1.set_title('Méthode du Coude')
            ax1.grid(True, alpha=0.3)
            
            kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans_final.fit_predict(X_scaled)
            
            pca_viz = PCA(n_components=2)
            X_viz = pca_viz.fit_transform(X_scaled)
            scatter = ax2.scatter(X_viz[:, 0], X_viz[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=70)
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            ax2.set_title('Clustering K-Means (K=3)')
            plt.colorbar(scatter, ax=ax2, label='Cluster')
            plt.tight_layout()
            results['clustering_plot'] = create_plot()
            
            df_temp = df.copy()
            df_temp['Cluster'] = clusters
            cluster_summary = df_temp.groupby('Cluster')[features + ['Rendement (t/ha)']].mean().round(2)
            results['cluster_table'] = cluster_summary.to_html(classes='table table-sm table-bordered', border=0)
        except:
            results['clustering_plot'] = ''
            results['cluster_table'] = '<p>Erreur</p>'
        
        # Matrice de corrélation
        try:
            plt.figure(figsize=(7, 6))
            corr = df[features + ['Rendement (t/ha)']].corr()
            sns.heatmap(corr, annot=True, cmap='YlGn', center=0, fmt='.2f', square=True)
            plt.title('Matrice de Corrélation', fontsize=14)
            plt.tight_layout()
            results['corr_plot'] = create_plot()
        except:
            results['corr_plot'] = ''
        
        return results
        
    except Exception as e:
        return {'error': 'analysis_failed', 'message': str(e)[:100], 'count': len(data)}

# ---------- GESTIONNAIRES D'ERREURS ----------
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_code=404, error_message="Page non trouvée"), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('error.html', error_code=500, error_message="Erreur interne"), 500

# ---------- ROUTES PRINCIPALES ----------
@app.route('/')
def index():
    cultures = ['Maïs', 'Cacao', 'Banane Plantain', 'Manioc', 'Tomate', 'Arachide', 'Coton']
    regions = ['Centre', 'Littoral', 'Ouest', 'Sud', 'Est', 'Nord', 'Extrême-Nord', 'Nord-Ouest', 'Sud-Ouest', 'Adamaoua']
    return render_template('index.html', cultures=cultures, regions=regions)

@app.route('/add', methods=['POST'])
def add_entry():
    try:
        data = {
            'type_culture': request.form.get('type_culture', ''),
            'surface_ha': float(request.form.get('surface_ha', 0)),
            'pluviometrie_mm': float(request.form.get('pluviometrie_mm', 0)),
            'engrais_kg': float(request.form.get('engrais_kg', 0)),
            'rendement_tonnes': float(request.form.get('rendement_tonnes', 0)),
            'prix_vente_fcfa': float(request.form.get('prix_vente_fcfa', 0) or 0)
        }
        region = request.form.get('region', 'Centre')
        
        errors = validate_input(data)
        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('index'))
        
        qualite = 'Faible' if data['rendement_tonnes'] < 5 else ('Moyenne' if data['rendement_tonnes'] < 12 else 'Bonne')
        
        new_entry = AgriData(
            region=region, type_culture=data['type_culture'],
            surface_ha=data['surface_ha'], pluviometrie_mm=data['pluviometrie_mm'],
            engrais_kg=data['engrais_kg'], rendement_tonnes=data['rendement_tonnes'], 
            qualite=qualite, prix_vente_fcfa=data['prix_vente_fcfa']
        )
        db.session.add(new_entry)
        db.session.commit()
        clear_cache()
        
        flash(f'✅ Parcelle enregistrée ! {data["rendement_tonnes"]} t/ha - Qualité {qualite}', 'success')
        
        # 🆕 Recommandations personnalisées
        personalized_recs = get_personalized_recommendation(new_entry)
        for rec in personalized_recs:
            flash(rec, 'info')
        
    except ValueError:
        flash('❌ Erreur : Veuillez saisir des nombres valides', 'error')
    except Exception as e:
        flash(f'❌ Erreur : {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    analysis = generate_full_analysis()
    return render_template('dashboard.html', analysis=analysis)

@app.route('/data')
def view_data():
    page = request.args.get('page', 1, type=int)
    per_page = 15
    
    query = AgriData.query
    region_filter = request.args.get('region', '')
    culture_filter = request.args.get('culture', '')
    qualite_filter = request.args.get('qualite', '')
    
    if region_filter:
        query = query.filter(AgriData.region == region_filter)
    if culture_filter:
        query = query.filter(AgriData.type_culture == culture_filter)
    if qualite_filter:
        query = query.filter(AgriData.qualite == qualite_filter)
    
    pagination = query.order_by(AgriData.date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    regions = db.session.query(AgriData.region).distinct().all()
    cultures = db.session.query(AgriData.type_culture).distinct().all()
    
    return render_template('data.html', pagination=pagination, 
                          regions=[r[0] for r in regions], cultures=[c[0] for c in cultures],
                          current_filters={'region': region_filter, 'culture': culture_filter, 'qualite': qualite_filter})

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_entry(id):
    entry = AgriData.query.get_or_404(id)
    cultures = ['Maïs', 'Cacao', 'Banane Plantain', 'Manioc', 'Tomate', 'Arachide', 'Coton']
    regions = ['Centre', 'Littoral', 'Ouest', 'Sud', 'Est', 'Nord', 'Extrême-Nord', 'Nord-Ouest', 'Sud-Ouest', 'Adamaoua']
    
    if request.method == 'POST':
        try:
            data = {
                'type_culture': request.form.get('type_culture', ''),
                'surface_ha': float(request.form.get('surface_ha', 0)),
                'pluviometrie_mm': float(request.form.get('pluviometrie_mm', 0)),
                'engrais_kg': float(request.form.get('engrais_kg', 0)),
                'rendement_tonnes': float(request.form.get('rendement_tonnes', 0)),
                'prix_vente_fcfa': float(request.form.get('prix_vente_fcfa', 0) or 0)
            }
            errors = validate_input(data)
            if errors:
                for error in errors:
                    flash(error, 'error')
                return render_template('edit.html', entry=entry, cultures=cultures, regions=regions)
            
            entry.region = request.form.get('region', 'Centre')
            entry.type_culture = data['type_culture']
            entry.surface_ha = data['surface_ha']
            entry.pluviometrie_mm = data['pluviometrie_mm']
            entry.engrais_kg = data['engrais_kg']
            entry.rendement_tonnes = data['rendement_tonnes']
            entry.prix_vente_fcfa = data['prix_vente_fcfa']
            entry.qualite = 'Faible' if data['rendement_tonnes'] < 5 else ('Moyenne' if data['rendement_tonnes'] < 12 else 'Bonne')
            
            db.session.commit()
            clear_cache()
            flash(f'✅ Entrée #{id} modifiée !', 'success')
            return redirect(url_for('view_data'))
        except ValueError:
            flash('❌ Erreur : nombres invalides', 'error')
    
    return render_template('edit.html', entry=entry, cultures=cultures, regions=regions)

@app.route('/delete/<int:id>')
def delete_entry(id):
    entry = AgriData.query.get_or_404(id)
    try:
        db.session.delete(entry)
        db.session.commit()
        clear_cache()
        flash(f'✅ Entrée #{id} supprimée !', 'success')
    except Exception as e:
        flash(f'❌ Erreur : {str(e)}', 'error')
    return redirect(url_for('view_data'))

@app.route('/export')
def export():
    csv_file = export_data_csv()
    if csv_file:
        return send_file(csv_file, as_attachment=True, download_name=os.path.basename(csv_file))
    flash('❌ Erreur lors de l\'export', 'error')
    return redirect(url_for('dashboard'))

@app.route('/backup')
def backup():
    success, backup_file = backup_database()
    if success:
        flash(f'✅ Sauvegarde effectuée', 'success')
    else:
        flash('❌ Erreur lors de la sauvegarde', 'error')
    return redirect(url_for('dashboard'))

@app.route('/health')
def health_check():
    health = {
        'status': 'OK',
        'database': 'Connectée',
        'records_count': AgriData.query.count(),
        'disk_usage': f"{psutil.disk_usage('/').percent}%" if PSUTIL_AVAILABLE else 'N/A',
        'memory_usage': f"{psutil.virtual_memory().percent}%" if PSUTIL_AVAILABLE else 'N/A',
        'python_version': platform.python_version(),
        'last_backup': 'Aucune'
    }
    backup_dir = 'backups'
    if os.path.exists(backup_dir):
        backups = sorted([f for f in os.listdir(backup_dir) if f.endswith('.db')])
        if backups:
            health['last_backup'] = backups[-1]
    return render_template('health.html', health=health)

@app.route('/about')
def about():
    return render_template('about.html')

# 🆕 CALENDRIER CULTURAL
@app.route('/calendar')
def cultural_calendar():
    culture = request.args.get('culture', '')
    info = CULTURAL_CALENDAR.get(culture, None) if culture else None
    return render_template('calendar.html', calendar=CULTURAL_CALENDAR, selected=culture, info=info)

# 🆕 GRAPHIQUES INTERACTIFS PLOTLY
@app.route('/interactive')
def interactive_plots():
    data = AgriData.query.all()
    if len(data) < 5:
        flash('⚠️ Pas assez de données pour les graphiques interactifs', 'warning')
        return redirect(url_for('dashboard'))
    
    df = pd.DataFrame([{
        'Culture': d.type_culture, 'Région': d.region,
        'Surface (ha)': d.surface_ha, 'Pluviométrie (mm)': d.pluviometrie_mm,
        'Engrais (kg/ha)': d.engrais_kg, 'Rendement (t/ha)': d.rendement_tonnes,
        'Qualité': d.qualite
    } for d in data])
    
    # Graphique 1 : Boxplot rendement par culture
    fig1 = px.box(df, x='Culture', y='Rendement (t/ha)', color='Culture',
                  title='Distribution des rendements par culture')
    plot1 = fig1.to_html(full_html=False)
    
    # Graphique 2 : Surface vs Rendement
    fig2 = px.scatter(df, x='Surface (ha)', y='Rendement (t/ha)', color='Qualité',
                      size='Engrais (kg/ha)', hover_data=['Culture', 'Région'],
                      title='Relation Surface vs Rendement')
    plot2 = fig2.to_html(full_html=False)
    
    # Graphique 3 : Rendement moyen par région
    region_avg = df.groupby('Région')['Rendement (t/ha)'].mean().reset_index()
    fig3 = px.bar(region_avg, x='Région', y='Rendement (t/ha)', color='Région',
                  title='Rendement moyen par région')
    plot3 = fig3.to_html(full_html=False)
    
    return render_template('interactive.html', plot1=plot1, plot2=plot2, plot3=plot3)

# ============================================================
# ROUTES API
# ============================================================

@app.route('/api/region-stats')
def region_stats():
    data = AgriData.query.all()
    if not data:
        return jsonify({'stats': {}})
    df = pd.DataFrame([{'region': d.region, 'rendement': d.rendement_tonnes} for d in data])
    stats = df.groupby('region')['rendement'].mean().round(2).to_dict()
    return jsonify({'stats': stats})

@app.route('/api/culture-stats')
def culture_stats():
    data = AgriData.query.all()
    if not data:
        return jsonify({'stats': {}})
    df = pd.DataFrame([{'culture': d.type_culture, 'rendement': d.rendement_tonnes} for d in data])
    stats = df.groupby('culture')['rendement'].mean().round(2).to_dict()
    return jsonify({'stats': stats})

@app.route('/api/trend')
def trend():
    data = AgriData.query.order_by(AgriData.date).all()
    if not data:
        return jsonify({'dates': [], 'rendements': []})
    df = pd.DataFrame([{'date': d.date, 'rendement': d.rendement_tonnes} for d in data])
    trend = df.groupby('date')['rendement'].mean().round(2)
    return jsonify({'dates': trend.index.tolist(), 'rendements': trend.values.tolist()})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        surface = float(data.get('surface', 5))
        pluie = float(data.get('pluie', 1200))
        engrais = float(data.get('engrais', 250))
        
        analysis = generate_full_analysis()
        if analysis and 'model_coef' in analysis:
            coef = analysis['model_coef']
            rendement = coef['intercept'] + coef['surface'] * surface + coef['pluie'] * pluie + coef['engrais'] * engrais
        else:
            rendement = 2.5 + 0.1 * surface + 0.002 * pluie + 0.01 * engrais
        
        rendement = max(0.5, round(rendement, 2))
        return jsonify({'rendement': rendement, 'total': round(rendement * surface, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/alerts')
def get_alerts():
    data = AgriData.query.all()
    alerts = []
    if data:
        df = pd.DataFrame([{'rendement': d.rendement_tonnes, 'pluie': d.pluviometrie_mm} for d in data])
        avg_rendement = df['rendement'].mean()
        avg_pluie = df['pluie'].mean()
        
        if avg_rendement < 5:
            alerts.append("⚠️ Rendement moyen faible. Vérifiez l'irrigation et les engrais.")
        elif avg_rendement > 15:
            alerts.append("🎉 Excellent rendement moyen ! Continuez vos pratiques.")
        
        if avg_pluie < 800:
            alerts.append("🌧️ Pluviométrie très faible. Irrigation fortement recommandée.")
    
    return jsonify({'alerts': alerts})

@app.route('/reset-demo')
def reset_demo():
    try:
        count_before = AgriData.query.count()
        AgriData.query.delete()
        db.session.commit()
        demo = generate_demo_data()
        for d in demo:
            db.session.add(d)
        db.session.commit()
        clear_cache()
        flash(f'✅ {count_before} données supprimées, {len(demo)} données de démo recréées !', 'success')
    except Exception as e:
        flash(f'❌ Erreur : {str(e)}', 'error')
    return redirect(url_for('dashboard'))

# ---------- LANCEMENT ----------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 70)
    print("🌾 AGRI-STAT - MON GENERAL")
    print("=" * 70)
    print(f"📍 Accueil        : http://127.0.0.1:{port}")
    print(f"📊 Dashboard      : http://127.0.0.1:{port}/dashboard")
    print(f"📋 Données        : http://127.0.0.1:{port}/data")
    print(f"📅 Calendrier     : http://127.0.0.1:{port}/calendar")
    print(f"📊 Interactif     : http://127.0.0.1:{port}/interactive")
    print(f"📄 À propos       : http://127.0.0.1:{port}/about")
    print("=" * 70)
    app.run(host='0.0.0.0', port=port, debug=False)
