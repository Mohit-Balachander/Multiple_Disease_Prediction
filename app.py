#app.py
import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Get working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load models
def load_model(version_folder, model_name):
    model_path = os.path.join(working_dir, "saved_models", version_folder, model_name)
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            # Verify it's actually a model and not a scaler
            if hasattr(model, 'predict'):
                return model
            else:
                st.error(f"Error: Loaded object from {model_path} is not a model (missing predict method)")
                return None
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {str(e)}")
        return None

# Function to load scaler if exists
def load_scaler(version_folder, scaler_name):
    try:
        scaler_path = os.path.join(working_dir, "saved_models", version_folder, scaler_name)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            # Verify it's actually a scaler
            if hasattr(scaler, 'transform'):
                return scaler
            else:
                st.error(f"Error: Loaded object from {scaler_path} is not a scaler")
                return None
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading scaler from {scaler_path}: {str(e)}")
        return None

# Function to load datasets
def load_dataset(dataset_name):
    dataset_path = os.path.join(working_dir, "dataset", dataset_name)
    return pd.read_csv(dataset_path)

# Function to evaluate model performance
def evaluate_model_performance(model, X_test, y_test, model_name):
    try:
        if model is None:
            st.error(f"Model {model_name} is None, cannot evaluate")
            return None
            
        y_pred = model.predict(X_test)
        
        # Ensure we have proper binary classification
        cm = confusion_matrix(y_test, y_pred)
        
        # Handle edge cases where classes might be missing
        if cm.shape != (2, 2):
            cm_full = np.zeros((2, 2), dtype=int)
            if cm.shape == (1, 1):
                if y_test.iloc[0] == 0:
                    cm_full[0, 0] = cm[0, 0]
                else:
                    cm_full[1, 1] = cm[0, 0]
            cm = cm_full
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
            'confusion_matrix': cm
        }
        return metrics
    except Exception as e:
        st.error(f"Error evaluating {model_name}: {str(e)}")
        return None

# Function to create confusion matrix heatmap
def create_confusion_matrix_plot(cm, title, disease_name):
    cm_array = np.array(cm)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        hoverongaps=False,
        colorscale='Blues',
        text=cm_array,
        texttemplate="%{text}",
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title=f'{title} - {disease_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=400,
        height=400,
        font=dict(size=12)
    )
    
    return fig

# Sidebar Navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                         ['Model Comparison Dashboard',
                          'Diabetes Prediction',
                          'Heart Disease Prediction',
                          'Parkinsons Prediction'],
                         menu_icon='hospital-fill',
                         icons=['bar-chart', 'activity', 'heart', 'person'],
                         default_index=0)

# ====== MODEL COMPARISON DASHBOARD ======
if selected == 'Model Comparison Dashboard':
    st.title('🔍 Enhanced Model Performance Comparison Dashboard')
    
    # Disease information with algorithm names
    diseases_info = {
        'Diabetes': {
            'model_file': 'diabetes_model.sav',
            'dataset': 'diabetes.csv',
            'target_col': 'Outcome',
            'scaler_old': None,
            'scaler_new': 'diabetes_scaler.sav',
            'old_algorithm': 'LogisticRegression',
            'new_algorithm': 'RandomForest'
        },
        'Heart': {
            'model_file': 'heart_disease_model.sav',
            'dataset': 'heart.csv',
            'target_col': 'target',
            'scaler_old': None,
            'scaler_new': 'heart_scaler.sav',
            'old_algorithm': 'DecisionTree',
            'new_algorithm': 'GradientBoosting'
        },
        'Parkinsons': {
            'model_file': 'parkinsons_model.sav',
            'dataset': 'parkinsons.csv',
            'target_col': 'status',
            'scaler_old': 'parkinsons_scaler.sav',
            'scaler_new': None,
            'old_algorithm': 'SVM',
            'new_algorithm': 'AdaBoost'
        }
    }
    
    # Load and evaluate all models
    all_metrics = []
    confusion_matrices = {}
    
    for disease, info in diseases_info.items():
        try:
            # Load dataset
            df = load_dataset(info['dataset'])
            
            # Prepare features and target
            if disease == 'Parkinsons':
                X = df.drop(columns=['name', info['target_col']], axis=1)
            else:
                X = df.drop(columns=info['target_col'], axis=1)
            y = df[info['target_col']]
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            
            # Evaluate Old Model
            try:
                old_model = load_model('old', info['model_file'])
                if old_model is not None:
                    X_test_old = X_test.copy()
                    
                    if info['scaler_old']:
                        scaler_old = load_scaler('old', info['scaler_old'])
                        if scaler_old:
                            X_test_old = scaler_old.transform(X_test_old)
                    
                    old_metrics = evaluate_model_performance(old_model, X_test_old, y_test, 
                                                           f'{disease}_{info["old_algorithm"]}')
                    if old_metrics:
                        old_metrics['model_type'] = 'Old'
                        old_metrics['disease'] = disease
                        old_metrics['algorithm'] = info['old_algorithm']
                        all_metrics.append(old_metrics)
                        confusion_matrices[f'{disease}_{info["old_algorithm"]}'] = old_metrics['confusion_matrix']
                
            except Exception as e:
                st.warning(f"Could not load old model for {disease}: {str(e)}")
            
            # Evaluate New Model
            try:
                new_model = load_model('new', info['model_file'])
                if new_model is not None:
                    X_test_new = X_test.copy()
                    
                    if info['scaler_new']:
                        scaler_new = load_scaler('new', info['scaler_new'])
                        if scaler_new:
                            X_test_new = scaler_new.transform(X_test_new)
                    
                    new_metrics = evaluate_model_performance(new_model, X_test_new, y_test, 
                                                           f'{disease}_{info["new_algorithm"]}')
                    if new_metrics:
                        new_metrics['model_type'] = 'New'
                        new_metrics['disease'] = disease
                        new_metrics['algorithm'] = info['new_algorithm']
                        all_metrics.append(new_metrics)
                        confusion_matrices[f'{disease}_{info["new_algorithm"]}'] = new_metrics['confusion_matrix']
                
            except Exception as e:
                st.warning(f"Could not load new model for {disease}: {str(e)}")
                
        except Exception as e:
            st.error(f"Error processing {disease}: {str(e)}")
    
    if all_metrics:
        # Create DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "🔄 Confusion Matrix", 
                                         "📈 Detailed Analysis", "🎯 Model Selection Guide"])
        
        with tab1:
            st.subheader("📊 Model Performance Comparison")
            
            fig = go.Figure()
            
            diseases = metrics_df['disease'].unique()
            
            for disease in diseases:
                disease_data = metrics_df[metrics_df['disease'] == disease]
                info = diseases_info[disease]
                
                old_data = disease_data[disease_data['model_type'] == 'Old']
                new_data = disease_data[disease_data['model_type'] == 'New']
                
                if not old_data.empty:
                    fig.add_trace(go.Bar(
                        name=f'{disease}_{info["old_algorithm"]}',
                        x=[f'{disease}_{info["old_algorithm"]}'],
                        y=[old_data.iloc[0]['accuracy']],
                        marker_color='lightblue',
                        text=[f"{old_data.iloc[0]['accuracy']:.3f}"],
                        textposition='auto',
                    ))
                
                if not new_data.empty:
                    fig.add_trace(go.Bar(
                        name=f'{disease}_{info["new_algorithm"]}',
                        x=[f'{disease}_{info["new_algorithm"]}'],
                        y=[new_data.iloc[0]['accuracy']],
                        marker_color='#FA8072',
                        text=[f"{new_data.iloc[0]['accuracy']:.3f}"],
                        textposition='auto',
                    ))
            
            fig.update_layout(
                title="Model Accuracy Comparison Across Diseases",
                xaxis_title="Models",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1]),
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics table
            st.subheader("📋 Detailed Performance Metrics")
            display_df = metrics_df[['disease', 'algorithm', 'accuracy', 'precision', 'recall', 'f1_score']].copy()
            display_df.columns = ['Disease', 'Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            display_df = display_df.round(4)
            
            st.dataframe(display_df, use_container_width=True)
        
        with tab2:
            st.subheader("🔄 Confusion Matrix Analysis")
            
            for disease in diseases:
                info = diseases_info[disease]
                st.markdown(f"### {disease} Disease Models")
                
                col1, col2 = st.columns(2)
                
                # Old model confusion matrix
                old_key = f'{disease}_{info["old_algorithm"]}'
                if old_key in confusion_matrices:
                    with col1:
                        fig_old = create_confusion_matrix_plot(
                            confusion_matrices[old_key], 
                            f'{info["old_algorithm"]} Model', 
                            disease
                        )
                        st.plotly_chart(fig_old, use_container_width=True)
                
                # New model confusion matrix
                new_key = f'{disease}_{info["new_algorithm"]}'
                if new_key in confusion_matrices:
                    with col2:
                        fig_new = create_confusion_matrix_plot(
                            confusion_matrices[new_key], 
                            f'{info["new_algorithm"]} Model', 
                            disease
                        )
                        st.plotly_chart(fig_new, use_container_width=True)
                
                # Performance comparison
                disease_metrics = metrics_df[metrics_df['disease'] == disease]
                if len(disease_metrics) == 2:
                    old_acc = disease_metrics[disease_metrics['model_type'] == 'Old']['accuracy'].iloc[0]
                    new_acc = disease_metrics[disease_metrics['model_type'] == 'New']['accuracy'].iloc[0]
                    improvement = new_acc - old_acc
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{info['old_algorithm']} Accuracy", f"{old_acc:.4f}")
                    with col2:
                        st.metric(f"{info['new_algorithm']} Accuracy", f"{new_acc:.4f}")
                    with col3:
                        st.metric("Improvement", f"{improvement:.4f}", f"{improvement:.4f}")
                
                st.markdown("---")
        
        with tab3:
            st.subheader("📈 Detailed Performance Analysis")
            
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in metrics_to_compare:
                st.markdown(f"#### {metric.replace('_', ' ').title()} Comparison")
                
                fig = go.Figure()
                
                for disease in diseases:
                    info = diseases_info[disease]
                    disease_data = metrics_df[metrics_df['disease'] == disease]
                    
                    old_data = disease_data[disease_data['model_type'] == 'Old']
                    new_data = disease_data[disease_data['model_type'] == 'New']
                    
                    if not old_data.empty:
                        fig.add_trace(go.Bar(
                            name=f'{disease}_{info["old_algorithm"]}',
                            x=[f'{disease}_{info["old_algorithm"]}'],
                            y=[old_data.iloc[0][metric]],
                            marker_color='lightblue',
                            text=[f"{old_data.iloc[0][metric]:.3f}"],
                            textposition='auto',
                            showlegend=False
                        ))
                    
                    if not new_data.empty:
                        fig.add_trace(go.Bar(
                            name=f'{disease}_{info["new_algorithm"]}',
                            x=[f'{disease}_{info["new_algorithm"]}'],
                            y=[new_data.iloc[0][metric]],
                            marker_color='#FA8072',
                            text=[f"{new_data.iloc[0][metric]:.3f}"],
                            textposition='auto',
                            showlegend=False
                        ))
                
                fig.update_layout(
                    title=f"{metric.replace('_', ' ').title()} Across All Models",
                    xaxis_title="Models",
                    yaxis_title=metric.replace('_', ' ').title(),
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance improvement analysis
            st.subheader("🚀 Performance Improvement Analysis")
            
            improvement_data = []
            for disease in diseases:
                info = diseases_info[disease]
                disease_data = metrics_df[metrics_df['disease'] == disease]
                if len(disease_data) == 2:
                    old_row = disease_data[disease_data['model_type'] == 'Old'].iloc[0]
                    new_row = disease_data[disease_data['model_type'] == 'New'].iloc[0]
                    
                    improvements = {
                        'Disease': disease,
                        'Algorithm Pair': f"{info['old_algorithm']} → {info['new_algorithm']}",
                        'Accuracy Improvement': new_row['accuracy'] - old_row['accuracy'],
                        'Precision Improvement': new_row['precision'] - old_row['precision'],
                        'Recall Improvement': new_row['recall'] - old_row['recall'],
                        'F1-Score Improvement': new_row['f1_score'] - old_row['f1_score']
                    }
                    improvement_data.append(improvements)
            
            if improvement_data:
                improvement_df = pd.DataFrame(improvement_data)
                
                # Create improvement visualization
                fig = go.Figure()
                
                for i, row in improvement_df.iterrows():
                    fig.add_trace(go.Scatter(
                        x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        y=[
                            row['Accuracy Improvement'],
                            row['Precision Improvement'],
                            row['Recall Improvement'],
                            row['F1-Score Improvement']
                        ],
                        mode='lines+markers',
                        name=row['Algorithm Pair'],
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title="Model Improvement Across Different Metrics",
                    xaxis_title="Metrics",
                    yaxis_title="Improvement (New - Old)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show improvement table
                st.dataframe(improvement_df.round(4), use_container_width=True)
        
        with tab4:
            st.subheader("🎯 Model Selection Recommendations")
            
            for disease in diseases:
                info = diseases_info[disease]
                disease_data = metrics_df[metrics_df['disease'] == disease]
                
                if len(disease_data) == 2:
                    st.markdown(f"### {disease} Disease Analysis")
                    
                    old_row = disease_data[disease_data['model_type'] == 'Old'].iloc[0]
                    new_row = disease_data[disease_data['model_type'] == 'New'].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            f"{info['old_algorithm']} Accuracy", 
                            f"{old_row['accuracy']:.4f}",
                            help="Traditional ML Algorithm"
                        )
                    
                    with col2:
                        st.metric(
                            f"{info['new_algorithm']} Accuracy", 
                            f"{new_row['accuracy']:.4f}",
                            help="Advanced ML Algorithm"
                        )
                    
                    with col3:
                        improvement = new_row['accuracy'] - old_row['accuracy']
                        st.metric(
                            "Accuracy Improvement", 
                            f"{improvement:+.4f}",
                            f"{(improvement/old_row['accuracy']*100):+.1f}%"
                        )
                    
                    with col4:
                        if improvement > 0.01:
                            st.success(f"✅ Use {info['new_algorithm']} Model")
                            recommendation = f"{info['new_algorithm']} shows significant improvement"
                        elif improvement < -0.01:
                            st.error(f"❌ Use {info['old_algorithm']} Model")
                            recommendation = f"{info['old_algorithm']} performs better"
                        else:
                            st.warning("⚖️ Similar Performance")
                            recommendation = "Both models perform similarly"
                    
                    # Detailed comparison
                    comparison_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        info['old_algorithm']: [old_row['accuracy'], old_row['precision'], 
                                              old_row['recall'], old_row['f1_score']],
                        info['new_algorithm']: [new_row['accuracy'], new_row['precision'], 
                                              new_row['recall'], new_row['f1_score']]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df['Difference'] = comparison_df[info['new_algorithm']] - comparison_df[info['old_algorithm']]
                    comparison_df = comparison_df.round(4)
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    st.info(f"💡 **Recommendation:** {recommendation}")
                    
                    st.markdown("---")
    
    else:
        st.error("❌ No model metrics could be loaded. Please check your model files and datasets.")

# Sidebar model version selector for predictions
if selected != 'Model Comparison Dashboard':
    st.sidebar.subheader("Select Model Version")
    model_version = st.sidebar.selectbox("Choose version:", ["old", "new"])

# ====== Diabetes Prediction ======
if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction using ML')
    
    # Load model
    diabetes_model = load_model(model_version, 'diabetes_model.sav')
    if diabetes_model is None:
        st.error("Failed to load diabetes prediction model. Please check the model file.")
        st.stop()
    
    # Load scaler if it exists
    diabetes_scaler = load_scaler(model_version, "diabetes_scaler.sav")

    # Show model info
    if model_version == "old":
        st.info("📊 Using: LogisticRegression")
    else:
        st.info("🧠 Using: RandomForest")

    col1, col2, col3 = st.columns(3)
    with col1: 
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    with col2: 
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=100)
    with col3: 
        BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
    with col1: 
        SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    with col2: 
        Insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
    with col3: 
        BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    with col1: 
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    with col2: 
        Age = st.number_input('Age', min_value=1, max_value=120, value=25)

    if st.button('🔍 Get Diabetes Prediction', type="primary"):
        user_input = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]]
        
        if diabetes_scaler:
            user_input = diabetes_scaler.transform(user_input)
        
        try:
            diab_prediction = diabetes_model.predict(user_input)
            diab_prob = diabetes_model.predict_proba(user_input) if hasattr(diabetes_model, 'predict_proba') else None
            
            if diab_prediction[0] == 1:
                st.error('⚠️ **The person is predicted to be diabetic**')
            else:
                st.success('✅ **The person is predicted to be non-diabetic**')
            
            if diab_prob is not None:
                confidence = max(diab_prob[0]) * 100
                st.info(f"🎯 **Prediction Confidence:** {confidence:.1f}%")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# ====== Heart Disease Prediction ======
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')
    
    # Load model
    heart_disease_model = load_model(model_version, 'heart_disease_model.sav')
    if heart_disease_model is None:
        st.error("Failed to load heart disease prediction model. Please check the model file.")
        st.stop()
    
    # Load scaler if it exists
    heart_scaler = load_scaler(model_version, "heart_scaler.sav")

    # Show model info
    if model_version == "old":
        st.info("🌳 Using: DecisionTree")
    else:
        st.info("🗳️ Using: GradientBoosting")

    col1, col2, col3 = st.columns(3)
    with col1: age = st.number_input('Age', min_value=1, max_value=120, value=50)
    with col2: sex = st.selectbox('Sex', [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    with col3: cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    with col1: trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, value=120)
    with col2: chol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    with col3: fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with col1: restecg = st.selectbox('Resting ECG', [0, 1, 2])
    with col2: thalach = st.number_input('Max Heart Rate', min_value=50, max_value=250, value=150)
    with col3: exang = st.selectbox('Exercise Induced Angina', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with col1: oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with col2: slope = st.selectbox('ST Slope', [0, 1, 2])
    with col3: ca = st.selectbox('Vessels Colored', [0, 1, 2, 3])
    with col1: thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

    if st.button('🔍 Get Heart Disease Prediction', type="primary"):
        user_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                      exang, oldpeak, slope, ca, thal]]
        
        if heart_scaler:
            user_input = heart_scaler.transform(user_input)
        
        try:
            heart_prediction = heart_disease_model.predict(user_input)
            heart_prob = heart_disease_model.predict_proba(user_input) if hasattr(heart_disease_model, 'predict_proba') else None
            
            if heart_prediction[0] == 1:
                st.error('⚠️ **The person is predicted to have heart disease**')
            else:
                st.success('✅ **The person is predicted to have no heart disease**')
            
            if heart_prob is not None:
                confidence = max(heart_prob[0]) * 100
                st.info(f"🎯 **Prediction Confidence:** {confidence:.1f}%")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# ====== Parkinson's Prediction ======
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")
    
    # Load model
    parkinsons_model = load_model(model_version, 'parkinsons_model.sav')
    if parkinsons_model is None:
        st.error("Failed to load Parkinson's prediction model. Please check the model file.")
        st.stop()
    
    # Load scaler
    parkinsons_scaler = load_scaler(model_version, "parkinsons_scaler.sav")

    # Show model info
    if model_version == "old":
        st.info("🔬 Using: SVM")
    else:
        st.info("⚡ Using: AdaBoost")

    st.warning("⚠️ **Note:** This prediction requires 22 voice measurement parameters. For demonstration, you can use the default values and modify a few key parameters.")
    
    # Initialize all input variables
    fo = fhi = flo = Jitter_percent = Jitter_Abs = RAP = PPQ = DDP = Shimmer = Shimmer_dB = None
    APQ3 = APQ5 = APQ = DDA = NHR = HNR = RPDE = DFA = spread1 = spread2 = D2 = PPE = None
    
    # Use expandable sections for better UX
    with st.expander("📊 Fundamental Frequency Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1: fo = st.number_input('MDVP:Fo(Hz)', value=119.992, step=0.001, format="%.3f")
        with col2: fhi = st.number_input('MDVP:Fhi(Hz)', value=157.302, step=0.001, format="%.3f")
        with col3: flo = st.number_input('MDVP:Flo(Hz)', value=74.997, step=0.001, format="%.3f")
    
    with st.expander("📈 Jitter Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1: Jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.00784, step=0.00001, format="%.5f")
        with col2: Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', value=0.00007, step=0.000001, format="%.6f")
        with col3: RAP = st.number_input('MDVP:RAP', value=0.00370, step=0.00001, format="%.5f")
        with col1: PPQ = st.number_input('MDVP:PPQ', value=0.00554, step=0.00001, format="%.5f")
        with col2: DDP = st.number_input('Jitter:DDP', value=0.01109, step=0.00001, format="%.5f")
    
    with st.expander("📉 Shimmer Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1: Shimmer = st.number_input('MDVP:Shimmer', value=0.04374, step=0.00001, format="%.5f")
        with col2: Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', value=0.426, step=0.001, format="%.3f")
        with col3: APQ3 = st.number_input('Shimmer:APQ3', value=0.02182, step=0.00001, format="%.5f")
        with col1: APQ5 = st.number_input('Shimmer:APQ5', value=0.03130, step=0.00001, format="%.5f")
        with col2: APQ = st.number_input('MDVP:APQ', value=0.02971, step=0.00001, format="%.5f")
        with col3: DDA = st.number_input('Shimmer:DDA', value=0.06545, step=0.00001, format="%.5f")
    
    with st.expander("🔊 Noise and Harmony Parameters"):
        col1, col2 = st.columns(2)
        with col1: NHR = st.number_input('NHR', value=0.02211, step=0.00001, format="%.5f")
        with col2: HNR = st.number_input('HNR', value=21.033, step=0.001, format="%.3f")
    
    with st.expander("📐 Nonlinear Dynamics Parameters"):
        col1, col2, col3, col4 = st.columns(4)
        with col1: RPDE = st.number_input('RPDE', value=0.414783, step=0.000001, format="%.6f")
        with col2: DFA = st.number_input('DFA', value=0.815285, step=0.000001, format="%.6f")
        with col3: spread1 = st.number_input('spread1', value=-4.813031, step=0.000001, format="%.6f")
        with col4: spread2 = st.number_input('spread2', value=0.266482, step=0.000001, format="%.6f")
    
    with st.expander("🧮 Additional Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1: D2 = st.number_input('D2', value=2.301442, step=0.000001, format="%.6f")
        with col2: PPE = st.number_input('PPE', value=0.284654, step=0.000001, format="%.6f")

    if st.button("🔍 Get Parkinson's Prediction", type="primary"):
        # Create user input array with all the collected values
        user_input = [[
            fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, 
            Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, 
            RPDE, DFA, spread1, spread2, D2, PPE
        ]]
        
        # Apply scaling if needed
        if parkinsons_scaler:
            user_input = parkinsons_scaler.transform(user_input)
        
        try:
            parkinsons_prediction = parkinsons_model.predict(user_input)
            parkinsons_prob = parkinsons_model.predict_proba(user_input) if hasattr(parkinsons_model, 'predict_proba') else None
            
            if parkinsons_prediction[0] == 1:
                st.error("⚠️ **The person is predicted to have Parkinson's disease**")
            else:
                st.success("✅ **The person is predicted to be healthy (No Parkinson's disease)**")
            
            if parkinsons_prob is not None:
                confidence = max(parkinsons_prob[0]) * 100
                st.info(f"🎯 **Prediction Confidence:** {confidence:.1f}%")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")