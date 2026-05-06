from flask import Flask, request, render_template
import pandas as pd
import os
import traceback

# =====================================================
# MATPLOTLIB FIX
# =====================================================

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# =====================================================
# SHAP
# =====================================================

import shap

# =====================================================
# CUSTOM MODULES
# =====================================================

from src.pipeline.predict_pipeline import CustomData
from src.utils import load_object

# =====================================================
# FLASK APP
# =====================================================

application = Flask(__name__)
app = application

# =====================================================
# HOME ROUTE
# =====================================================

@app.route('/')
def index():
    return render_template('index.html')

# =====================================================
# PREDICTION ROUTE
# =====================================================

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    else:

        try:

            # =================================================
            # GET FORM DATA
            # =================================================

            gender = request.form.get('gender')

            race_ethnicity = request.form.get('ethnicity')

            parental_level_of_education = request.form.get(
                'parental_level_of_education'
            )

            lunch = request.form.get('lunch')

            test_preparation_course = request.form.get(
                'test_preparation_course'
            )

            reading_score = float(
                request.form.get('reading_score')
            )

            writing_score = float(
                request.form.get('writing_score')
            )

            # =================================================
            # CREATE CUSTOM DATA OBJECT
            # =================================================

            data = CustomData(

                gender=gender,

                race_ethnicity=race_ethnicity,

                parental_level_of_education=parental_level_of_education,

                lunch=lunch,

                test_preparation_course=test_preparation_course,

                reading_score=reading_score,

                writing_score=writing_score
            )

            # =================================================
            # CONVERT TO DATAFRAME
            # =================================================

            pred_df = data.get_data_as_data_frame()

            print(pred_df)

            # ORIGINAL FEATURE NAMES
            original_feature_names = pred_df.columns

            # =================================================
            # LOAD MODEL + PREPROCESSOR
            # =================================================

            model = load_object("artifacts/model.pkl")

            preprocessor = load_object("artifacts/preprocessor.pkl")

            # =================================================
            # TRANSFORM INPUT DATA
            # =================================================

            scaled_data = preprocessor.transform(pred_df)

            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(scaled_data)

            predicted_score = round(prediction[0], 2)

            # =================================================
            # PERFORMANCE CATEGORY
            # =================================================

            if predicted_score >= 90:
                performance = "Excellent"

            elif predicted_score >= 75:
                performance = "Good"

            elif predicted_score >= 50:
                performance = "Average"

            else:
                performance = "Weak"

            # =================================================
            # SHAP EXPLAINABLE AI
            # =================================================

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(scaled_data)

            # =================================================
            # CREATE STATIC FOLDER
            # =================================================

            os.makedirs("static", exist_ok=True)

            # =================================================
            # PLOT PATH
            # =================================================

            plot_path = os.path.join(
                "static",
                "shap_plot.png"
            )

            # =================================================
            # SHAP BAR PLOT
            # =================================================

            # =====================================================
            # CREATE FEATURE IMPORTANCE BAR GRAPH
            # =====================================================
            feature_importance = abs(shap_values[0])
            # Take top features
            top_indices = feature_importance.argsort()[-10:]

            top_feature_names = [
            str(original_feature_names[i % len(original_feature_names)])
            for i in top_indices
            ]

            top_feature_values = feature_importance[top_indices]

            # Create plot
            plt.figure(figsize=(12, 6))

            plt.barh(
                top_feature_names,
                top_feature_values
                )

            plt.xlabel("SHAP Importance")
            plt.ylabel("Features")
            plt.title("Top Feature Importance")

            # Save plot
            plt.savefig(
                plot_path,
                bbox_inches='tight'
                )
            plt.close()

            # =================================================
            # FEATURE IMPORTANCE TABLE
            # =================================================

            # =====================================================
            # FEATURE IMPORTANCE TABLE
            # =====================================================

            shap_contributions = abs(
                shap_values[0][:len(original_feature_names)]
                )
            importance_data = list(
                zip(
                    original_feature_names,
                    shap_contributions
                    )
            )
            importance_data = sorted(
                importance_data,
                key=lambda x: x[1],
                reverse=True
            )
            top_features = importance_data[:5]

            # =================================================
            # RECOMMENDATIONS
            # =================================================

            recommendations = []

            if reading_score < 60:
                recommendations.append(
                    "Improve reading practice daily."
                )

            if writing_score < 60:
                recommendations.append(
                    "Focus more on writing exercises."
                )

            if test_preparation_course == "none":
                recommendations.append(
                    "Complete a test preparation course."
                )

            if predicted_score < 50:
                recommendations.append(
                    "Student is at academic risk. Extra mentoring recommended."
                )

            if predicted_score >= 80:
                recommendations.append(
                    "Excellent performance predicted. Maintain consistency."
                )

            if predicted_score >= 60 and predicted_score < 80:
                recommendations.append(
                    "Good performance predicted. Aim for excellence."
                )

            if predicted_score >= 50 and predicted_score < 60:
                recommendations.append(
                    "Average performance predicted. Focus on weak areas."
                )
            # =================================================
            # RETURN TO HTML
            # =================================================

            return render_template(

                'home.html',

                results=predicted_score,

                performance=performance,

                shap_plot=plot_path,

                importance_data=top_features,

                recommendations=recommendations,

                gender=gender,

                race_ethnicity=race_ethnicity,

                parental_level_of_education=parental_level_of_education,

                lunch=lunch,

                test_preparation_course=test_preparation_course,

                reading_score=reading_score,

                writing_score=writing_score
            )

        except Exception as e:

            traceback.print_exc()

            return render_template(
                'home.html',
                results="Error occurred"
            )

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )