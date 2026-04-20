import os
import io
import base64
import warnings
import logging
import time
import json
import shutil
import platform
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# Essayer d'importer psutil (optionnel)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

warnings.filterwarnings('ignore')

# ---------- CONFIGURATION DES LOGS (FIABILITÉ) ----------
logging.basicConfig(
    filename='agristat.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def log_action(action, details=""):
    """Enregistre les actions importantes"""
    logging.info(f"{action} - {details}")
    print(f"📝 LOG: {action} - {details}")

# ---------- INITIALISATION FLASK ----------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'agristat_secret_key_2026_ultra_secure'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agristat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- SYSTÈME DE CACHE (EFFICACITÉ) ----------
_plot_cache = {}
_cache_time = {}
CACHE_DURATION = 300

def get_cached_plot(key, generate_func):
    current_time = time.time()
    if key in _plot_cache and (current_time - _cache_time.get(key, 0) < CACHE_DURATION):
        log_action("CACHE_HIT", key)
        return _plot_cache[key]
    log_action("CACHE_MISS", key)
    plot = generate_func()
    _plot_cache[key] = plot
    _cache_time[key] = current_time
    return plot

def clear_cache():
    global _plot_cache, _cache_time
    _plot_cache.clear()
    _cache_time.clear()
    log_action("CACHE_CLEAR", "Cache vidé")

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

# ---------- VALIDATION ROBUSTE DES DONNÉES ----------
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

# ---------- SAUVEGARDE ET RESTAURATION (FIABILITÉ) ----------
def backup_database():
    try:
        backup_dir = 'backups'
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{backup_dir}/agristat_backup_{timestamp}.db"
        shutil.copy('agristat.db', backup_file)
        backups = sorted([f for f in os.listdir(backup_dir) if f.endswith('.db')])
        if len(backups) > 10:
            for old_backup in backups[:-10]:
                os.remove(os.path.join(backup_dir, old_backup))
        log_action("BACKUP", f"Sauvegarde créée : {backup_file}")
        return True, backup_file
    except Exception as e:
        log_action("BACKUP_ERROR", str(e))
        return False, None

def export_data_csv():
    try:
        data = AgriData.query.all()
        df = pd.DataFrame([{
            'date': d.date, 'region': d.region, 'culture': d.type_culture,
            'surface_ha': d.surface_ha, 'pluviometrie_mm': d.pluviometrie_mm,
            'engrais_kg': d.engrais_kg, 'rendement_tonnes': d.rendement_tonnes, 'qualite': d.qualite
        } for d in data])
        export_dir = 'exports'
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"{export_dir}/agristat_export_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        log_action("EXPORT", f"Données exportées : {csv_file}")
        return csv_file
    except Exception as e:
        log_action("EXPORT_ERROR", str(e))
        return None

# ---------- GÉNÉRATION DES DONNÉES DE DÉMO ----------
def generate_demo_data():
    np.random.seed(42)
    demo_data = []
    cultures = ['Maïs', 'Cacao', 'Banane Plantain', 'Manioc', 'Tomate']
    regions = ['Centre', 'Littoral', 'Ouest', 'Sud', 'Est']
    
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
        
        demo_data.append(AgriData(
            region=region, type_culture=culture,
            surface_ha=round(surface, 2), pluviometrie_mm=round(pluie, 1),
            engrais_kg=round(engrais, 1), rendement_tonnes=round(rendement, 2), qualite=qualite
        ))
    return demo_data

# ---------- CRÉATION DE LA BASE DE DONNÉES ----------
with app.app_context():
    db.create_all()
    if AgriData.query.count() == 0:
        print("🌾 Génération des données agricoles de démonstration...")
        demo = generate_demo_data()
        for d in demo:
            db.session.add(d)
        db.session.commit()
        print(f"✅ {len(demo)} parcelles ajoutées")
        log_action("INIT", f"{len(demo)} données démo créées")

# ---------- FONCTION POUR CONVERTIR GRAPHIQUE EN BASE64 ----------
def create_plot():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120, facecolor='white')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

# ---------- GÉNÉRATION DE RECOMMANDATIONS INTELLIGENTES ----------
def generate_recommendations(df):
    recommendations = []
    avg_rendement = df['Rendement (t/ha)'].mean()
    avg_pluie = df['Pluviométrie (mm)'].mean()
    avg_engrais = df['Engrais (kg/ha)'].mean()
    
    if avg_pluie < 1000:
        recommendations.append({'icon': '💧', 'title': 'Irrigation recommandée', 'desc': f'Pluviométrie moyenne ({avg_pluie:.0f} mm) inférieure au seuil optimal. Envisagez l\'irrigation.', 'color': 'info'})
    else:
        recommendations.append({'icon': '✅', 'title': 'Pluviométrie satisfaisante', 'desc': f'Pluviométrie moyenne ({avg_pluie:.0f} mm) dans la norme.', 'color': 'success'})
    
    if avg_engrais < 200:
        recommendations.append({'icon': '🧪', 'title': 'Augmenter les intrants', 'desc': f'Utilisation d\'engrais ({avg_engrais:.0f} kg/ha) inférieure à la moyenne. Augmentez progressivement.', 'color': 'warning'})
    elif avg_engrais > 500:
        recommendations.append({'icon': '⚠️', 'title': 'Surplus d\'engrais', 'desc': f'Utilisation d\'engrais ({avg_engrais:.0f} kg/ha) élevée. Risque de pollution et coûts inutiles.', 'color': 'danger'})
    else:
        recommendations.append({'icon': '✅', 'title': 'Gestion d\'engrais optimale', 'desc': f'Utilisation d\'engrais ({avg_engrais:.0f} kg/ha) dans la norme.', 'color': 'success'})
    
    small_farms = df[df['Surface (ha)'] < 2]
    if len(small_farms) > len(df) * 0.3:
        recommendations.append({'icon': '🤝', 'title': 'Coopérative conseillée', 'desc': 'Les petites exploitations pourraient bénéficier d\'une mise en commun des ressources.', 'color': 'info'})
    
    best_culture = df.groupby('Culture')['Rendement (t/ha)'].mean().idxmax()
    best_rendement = df.groupby('Culture')['Rendement (t/ha)'].mean().max()
    recommendations.append({'icon': '🏆', 'title': f'Culture star : {best_culture}', 'desc': f'Rendement moyen de {best_rendement:.1f} t/ha. À privilégier dans vos conditions.', 'color': 'success'})
    
    return recommendations

# ---------- FONCTION D'ANALYSE COMPLÈTE ----------
def generate_full_analysis():
    data = AgriData.query.all()
    if len(data) < 10:
        return None
    
    df_data = []
    for d in data:
        df_data.append({
            'Surface (ha)': d.surface_ha, 'Pluviométrie (mm)': d.pluviometrie_mm,
            'Engrais (kg/ha)': d.engrais_kg, 'Rendement (t/ha)': d.rendement_tonnes,
            'Qualité': d.qualite, 'Culture': d.type_culture, 'Région': d.region
        })
    df = pd.DataFrame(df_data)
    
    results = {}
    results['count'] = len(df)
    results['cultures'] = df['Culture'].unique().tolist()
    results['regions'] = df['Région'].unique().tolist()
    
    # Recommandations
    results['recommendations'] = generate_recommendations(df)
    
    # Stats descriptives
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
        plt.title('Module 1 : Régression Simple\nRendement = f(Pluviométrie)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        results['reg_simple_plot'] = create_plot()
        results['reg_simple_eq'] = f"Rendement = {model_simple.intercept_:.2f} + {model_simple.coef_[0]:.4f} × Pluviométrie"
        results['reg_simple_r2'] = round(r2_simple, 3)
    except Exception as e:
        results['reg_simple_plot'] = ''
        results['reg_simple_eq'] = f'Erreur: {str(e)[:50]}'
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
        plt.plot([min_v, max_v], [min_v, max_v], '#FF8F00', linestyle='--', linewidth=2, label='Prédiction parfaite')
        plt.xlabel('Rendement réel (t/ha)', fontsize=12)
        plt.ylabel('Rendement prédit (t/ha)', fontsize=12)
        plt.title(f'Module 2 : Régression Multiple\nR²={r2_multi:.3f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        results['reg_multi_plot'] = create_plot()
        coef = model_multi.coef_
        results['reg_multi_eq'] = f"Rendement = {model_multi.intercept_:.2f} + {coef[0]:.3f}×Surface + {coef[1]:.4f}×Pluie + {coef[2]:.3f}×Engrais"
        results['reg_multi_r2'] = round(r2_multi, 3)
    except Exception as e:
        results['reg_multi_plot'] = ''
        results['reg_multi_eq'] = f'Erreur: {str(e)[:50]}'
        results['reg_multi_r2'] = 0
    
    # Standardisation pour les modules suivants
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
    except Exception as e:
        results['pca_plot'] = ''
        results['pca_var'] = 0
    
    # Module 4 : Classification supervisée
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
    except Exception as e:
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
    except Exception as e:
        results['clustering_plot'] = ''
        results['cluster_table'] = f'<p>Erreur: {str(e)[:50]}</p>'
    
    # Matrice de corrélation
    try:
        plt.figure(figsize=(7, 6))
        corr = df[features + ['Rendement (t/ha)']].corr()
        sns.heatmap(corr, annot=True, cmap='YlGn', center=0, fmt='.2f', square=True)
        plt.title('Matrice de Corrélation', fontsize=14)
        plt.tight_layout()
        results['corr_plot'] = create_plot()
    except Exception as e:
        results['corr_plot'] = ''
    
    return results

# ---------- GESTIONNAIRES D'ERREURS (ROBUSTESSE) ----------
@app.errorhandler(404)
def not_found_error(error):
    log_action("ERROR_404", request.url)
    return render_template('error.html', error_code=404, error_message="Page non trouvée", suggestion="La page que vous cherchez n'existe pas."), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    log_action("ERROR_500", str(error))
    return render_template('error.html', error_code=500, error_message="Erreur interne du serveur", suggestion="Réessayez dans quelques instants."), 500

# ---------- ROUTES ----------
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
            'rendement_tonnes': float(request.form.get('rendement_tonnes', 0))
        }
        region = request.form.get('region', 'Centre')
        
        errors = validate_input(data)
        if errors:
            for error in errors:
                flash(error, 'error')
            log_action("VALIDATION_ERROR", str(errors))
            return redirect(url_for('index'))
        
        qualite = 'Faible' if data['rendement_tonnes'] < 5 else ('Moyenne' if data['rendement_tonnes'] < 12 else 'Bonne')
        
        new_entry = AgriData(
            region=region, type_culture=data['type_culture'],
            surface_ha=data['surface_ha'], pluviometrie_mm=data['pluviometrie_mm'],
            engrais_kg=data['engrais_kg'], rendement_tonnes=data['rendement_tonnes'], qualite=qualite
        )
        db.session.add(new_entry)
        db.session.commit()
        clear_cache()
        
        flash(f'✅ Parcelle enregistrée ! {data["rendement_tonnes"]} t/ha - Qualité {qualite}', 'success')
        log_action("ADD_SUCCESS", f"Culture={data['type_culture']}, Rendement={data['rendement_tonnes']}")
        
    except ValueError:
        flash('❌ Erreur : Veuillez saisir des nombres valides', 'error')
        log_action("VALUE_ERROR", "Conversion échouée")
    except Exception as e:
        flash(f'❌ Erreur inattendue : {str(e)[:100]}', 'error')
        log_action("UNEXPECTED_ERROR", str(e))
    
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    analysis = generate_full_analysis()
    if analysis is None:
        flash('⚠️ Pas assez de données (minimum 10 parcelles). Ajoutez des données.', 'warning')
    return render_template('dashboard.html', analysis=analysis)

@app.route('/data')
def view_data():
    page = request.args.get('page', 1, type=int)
    per_page = 15
    pagination = AgriData.query.order_by(AgriData.date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('data.html', pagination=pagination)

@app.route('/export')
def export():
    csv_file = export_data_csv()
    if csv_file:
        flash(f'✅ Données exportées avec succès !', 'success')
        return send_file(csv_file, as_attachment=True, download_name=os.path.basename(csv_file))
    else:
        flash('❌ Erreur lors de l\'export', 'error')
        return redirect(url_for('dashboard'))

@app.route('/backup')
def backup():
    success, backup_file = backup_database()
    if success:
        flash(f'✅ Sauvegarde effectuée : {os.path.basename(backup_file)}', 'success')
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

@app.route('/stats')
def stats():
    total = AgriData.query.count()
    cultures = db.session.query(AgriData.type_culture, func.count(AgriData.id)).group_by(AgriData.type_culture).all()
    regions = db.session.query(AgriData.region, func.count(AgriData.id)).group_by(AgriData.region).all()
    avg_rendement = db.session.query(func.avg(AgriData.rendement_tonnes)).scalar()
    return render_template('stats.html', total=total, cultures=cultures, regions=regions, avg_rendement=avg_rendement)

# ---------- LANCEMENT ----------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 70)
    print("🌾 AGRI-STAT - Analyse des Rendements Agricoles")
    print("=" * 70)
    print(f" Accueil        : http://127.0.0.1:{port}")
    print(f" Dashboard      : http://127.0.0.1:{port}/dashboard")
    print(f" Données        : http://127.0.0.1:{port}/data")
    print(f" Sauvegarde     : http://127.0.0.1:{port}/backup")
    print(f" Export CSV     : http://127.0.0.1:{port}/export")
    print(f"Santé système  : http://127.0.0.1:{port}/health")
    print("=" * 70)
    log_action("START", f"Application lancée sur le port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
