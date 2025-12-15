import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*version.*')

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import pickle
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Use Agg backend for headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load Model, Top Features, Encoder, and Scaler
with open(BASE_DIR / "catboost_tuned_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "top_features.pkl", "rb") as f:
    top_features = pickle.load(f)

with open(BASE_DIR / "categorical_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open(BASE_DIR / "scaler_stepwise_features.pkl", "rb") as f:
    scaler = pickle.load(f)

# Form categorical features
form_categorical_features = ['Final Grade', 'County', 'Sub_County', 'Location', 'Course']

# Correct encoder mapping based on actual encoder contents
cat_to_encoder_idx = {
    'County': 2,
    'Sub_County': 3,
    'Location': 4,
    'Course': 7,
    'Final Grade': 9
}

# Load counties data
with open(BASE_DIR / "kenyan_counties.json", "r") as f:
    counties_data = json.load(f)

# Create county-subcounty mapping
county_subcounty_map = {county['name']: county['sub_counties'] for county in counties_data}

# Update categorical values to use counties from JSON
categorical_values = {feat: encoders[idx].classes_.tolist() for feat, idx in cat_to_encoder_idx.items()}
categorical_values['County'] = list(county_subcounty_map.keys())

# Create ordered features list with type information and categorical values
features_info = []
for feat in top_features:
    feature_dict = {
        'name': feat, 
        'type': 'categorical' if feat in form_categorical_features else 'numerical'
    }
    # Add categorical values if it's a categorical feature
    if feat in form_categorical_features and feat in categorical_values:
        feature_dict['values'] = categorical_values[feat]
    features_info.append(feature_dict)


# ------------------------------
# Analysis helpers
# ------------------------------
_analysis_cache = None


def _load_and_clean_dataset():
    """Load and clean the analysis dataset with the provided rules."""
    global _analysis_cache
    if _analysis_cache is not None:
        return _analysis_cache

    csv_path = BASE_DIR / "All Graduated spreadsheet - Sheet1 (1) - All Graduated spreadsheet - Sheet1 (1).csv"
    if not csv_path.exists():
        _analysis_cache = (None, {"error": f"Dataset not found at {csv_path.name}"})
        return _analysis_cache

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        _analysis_cache = (None, {"error": f"Failed to load CSV: {exc}"})
        return _analysis_cache

    # Numeric columns: fill with mean then ensure numeric type
    numeric_cols = [
        "Height", "Weight", "Theory Exam - Icare", "Practical Scores /80 - Icare",
        "Viva Scores /20 - Icare", "Hospital Internship Score", "Final Score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].mean(), inplace=True)

    # Categorical columns: fill with mode or sensible defaults
    cat_cols = [
        "Sub_County", "Location", "Hospital Internship Attended",
        "Final Grade", "Start Date of Class", "End Date of Class"
    ]
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()
            fallback = mode_val.iloc[0] if not mode_val.empty else ""
            df[col].fillna(fallback, inplace=True)

    if "Education_Level" in df.columns:
        df["Education_Level"].fillna("High School", inplace=True)
    if "Primary/Highschool_Grade" in df.columns:
        df["Primary/Highschool_Grade"].fillna("D", inplace=True)

    # Parse dates
    for col in ["Start Date of Class", "End Date of Class"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
        df["Year of Birth"] = df["DOB"].dt.year

    # Rename and normalize year column
    if "Yearof Class" in df.columns:
        df["Yearof Class"] = df["Yearof Class"].fillna(2025)
        df["Yearof Class"] = df["Yearof Class"].astype(int, errors="ignore")
        df.rename(columns={"Yearof Class": "Year of Class"}, inplace=True)
    if "Year of Class" in df.columns:
        df["Year of Class"] = df["Year of Class"].fillna(2025)
        df["Year of Class"] = df["Year of Class"].astype(int, errors="ignore")

    # Remove exam rows
    if "Class ID" in df.columns:
        df = df[~df["Class ID"].str.startswith("Exam", na=False)]

    # Attendance averages
    attendance_columns = [
        "Attendance(Week 1)", "Attendance(Week 2)", "Attendance(Week 3)",
        "Attendance(Week 4)", "Attendance(Week 5)", "Attendance(Week 6)",
        "Attendance(Week 7)",
    ]
    existing_att_cols = [c for c in attendance_columns if c in df.columns]
    for col in existing_att_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if existing_att_cols:
        df["Average Attendance"] = df[existing_att_cols].mean(axis=1).round(0)
        df["Average Attendance"].fillna(df["Average Attendance"].mean(), inplace=True)

    # CAT averages
    cat_columns = ['CAT 1', 'CAT 2', 'CAT 3', 'CAT 4', 'CAT 5', 'CAT 6']
    existing_cat_cols = [c for c in cat_columns if c in df.columns]
    for col in existing_cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if existing_cat_cols and "Class ID" in df.columns:
        df["Average Cats"] = np.where(
            df["Class ID"].str.contains("Childcare", case=False, na=False),
            (df[existing_cat_cols[:4]].mean(axis=1)).round(0) if len(existing_cat_cols) >= 4 else np.nan,
            (df[existing_cat_cols].mean(axis=1)).round(0),
        )

    # Handout averages
    handout_columns = [
        'Handout 1 Score', 'Handout 2 Score', 'Handout 3 Score', 'Handout 4 Score',
        'Handout 5 Score', 'Handout 6 Score', 'Handout 7 Score', 'Handout 8 Score',
        'Handout 9 Score', 'Handout 10 score'
    ]
    existing_handout_cols = [c for c in handout_columns if c in df.columns]
    for col in existing_handout_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if existing_handout_cols and "Class ID" in df.columns:
        df["Average Handouts"] = np.where(
            df["Class ID"].str.contains("Childcare", case=False, na=False),
            0,
            (df[existing_handout_cols].mean(axis=1)).round(0),
        )

    # Blood sugar normalization
    if "Blood Sugar Test" in df.columns:
        df["Blood Sugar Test"] = pd.to_numeric(df["Blood Sugar Test"], errors="coerce")
        df["Blood Sugar Test"].replace(33, 3, inplace=True)

    # Practical scores
    practical_columns = [
        'Hand Hygiene', 'Gloving', 'Denture Cleaning', 'Oral Care', 'Oral Care(Unconscious)',
        'Hand and Feet Care', 'Bed Making', 'Positioning', 'Back Care', 'ROM', 'Wheel Chair Transfer',
        'Wound Care', 'Sponge Bath', 'Perineal Care', 'Hair Care', 'Fall Prevention', 'Dressing',
        'Tube Feeding', 'Daiper Change', 'Catheter Care', 'Bed Pan', 'Vitals', 'Giving Oxygen',
        'Blood Sugar Test', 'Insulin Injection', 'Administrating Meds', 'CPR',
        'Use of Medical Equipment', 'Shaving'
    ]
    childcare_practicals = [
        'Hand Hygiene', 'Gloving', 'Bed Making', 'Sponge Bath', 'Daiper Change', 'Vitals',
        'Administrating Meds', 'Paediatric Advanced Life Support(PALS)', 'Toilet Assistance',
        'Dressing the Baby', 'Using the Stroller', 'Supporting the Baby in Walker',
        'Latching the Baby', 'Care of the Ear, Nose and Nails of the Baby', 'Feeding'
    ]
    existing_prac_cols = [c for c in practical_columns if c in df.columns]
    for col in existing_prac_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Class ID" in df.columns:
        df["Average Practicals"] = np.where(
            df["Class ID"].str.contains("Childcare", case=False, na=False),
            ((df[[c for c in childcare_practicals if c in df.columns]].sum(axis=1) / 45) * 100).round(0),
            ((df[[c for c in practical_columns if c in df.columns]].sum(axis=1) / 87) * 100).round(0)
        )
        df["Average Practicals"] = df["Average Practicals"].fillna(0).astype(int, errors="ignore")

    # Entry test and attendance fill
    if "Entry Test" in df.columns:
        df["Entry Test"] = pd.to_numeric(df["Entry Test"], errors="coerce")
        df["Entry Test"].fillna(df["Entry Test"].mean(), inplace=True)

    # Practical-Icare
    if {"Practical Scores /80 - Icare", "Viva Scores /20 - Icare"}.issubset(df.columns):
        df["Practical Scores /80 - Icare"] = pd.to_numeric(df["Practical Scores /80 - Icare"], errors="coerce")
        df["Viva Scores /20 - Icare"] = pd.to_numeric(df["Viva Scores /20 - Icare"], errors="coerce")
        df["Practical-Icare"] = df["Practical Scores /80 - Icare"] + df["Viva Scores /20 - Icare"]

    # Course duration
    if {"End Date of Class", "Start Date of Class"}.issubset(df.columns):
        df["Course Duration"] = (df["End Date of Class"] - df["Start Date of Class"]).dt.days
        df["Course Duration Months"] = ((df["Course Duration"] / 30).round()).astype(int, errors="ignore")

    # Drop unwanted columns if present
    drop_cols = [
        'Hand Hygiene','Gloving','Denture Cleaning','Oral Care','Oral Care(Unconscious)',
        'Hand and Feet Care','Bed Making','Positioning','Back Care','ROM','Wheel Chair Transfer',
        'Wound Care','Sponge Bath','Perineal Care','Hair Care','Fall Prevention','Dressing',
        'Tube Feeding','Daiper Change','Catheter Care','Bed Pan','Vitals','Giving Oxygen',
        'Blood Sugar Test','Insulin Injection','Administrating Meds','CPR',
        'Use of Medical Equipment','Shaving', 'Handout 1 Score',
        'Handout 2 Score','Handout 3 Score','Handout 4 Score',
        'Handout 5 Score','Handout 6 Score','Handout 7 Score','Handout 8 Score',
        'Handout 9 Score','Handout 10 score', 'CAT 1','CAT 2','CAT 3',
        'CAT 4','CAT 5','CAT 6', 'Attendance(Week 1)', 'Attendance(Week 2)',
        'Attendance(Week 3)', 'Attendance(Week 4)', 'Attendance(Week 5)',
        'Attendance(Week 6)', 'Attendance(Week 7)',
        'Paediatric Advanced Life Support(PALS)',
        'Toilet Assistance', 'Practical Scores /80 - Icare', 'Viva Scores /20 - Icare',
        'Dressing the Baby', 'Using the Stroller', 'Supporting the Baby in Walker',
        'Latching the Baby', 'Care of the Ear, Nose and Nails of the Baby',
        'Feeding','End Date of Class','Start Date of Class','Year of Birth','DOB',
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    if existing_drop_cols:
        df.drop(columns=existing_drop_cols, inplace=True)

    # Replacements
    if "Location" in df.columns:
        df["Location"].replace({"Null": "Kangemi"}, inplace=True)
    if "County" in df.columns:
        df["County"].replace({"Null": "Nairobi"}, inplace=True)
    if "Final Grade" in df.columns:
        df["Final Grade"].replace(
            {
                "FAIL": "Fail",
                "DISTINCTION": "Distinction",
                "PASS": "Pass",
                "CREDIT": "Credit",
                "HIGHER CREDIT": "Higher Credit",
            },
            inplace=True,
        )

    # Derive Course from Class ID
    if "Class ID" in df.columns:
        df["Course"] = df["Class ID"].apply(
            lambda x: "Childcare" if "Childcare" in str(x) else "Eldercare"
        )

    # Drop columns if present
    drop_cols = [
        "Similar", "Status", "First Name", "Middle Name", "Last Name",
        "Full name", "Reg No", "UFA Reg No"
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df.drop(columns=existing_drop, inplace=True)

    # Transition normalization and defaults
    if "Transitioned_to_Caregiving" in df.columns:
        df["Transitioned_to_Caregiving"].fillna("Didn't Transition", inplace=True)
        df["Transitioned_to_Caregiving"].replace(
            ["Transitioned_to_Caregiving", "Transitioned to care"],
            "Transitioned to Care",
            inplace=True,
        )
        df["Transitioned_to_Caregiving"].replace(
            {"Not Found": "Didn't Transition"}, inplace=True
        )

    # Education level normalization
    if "Education_Level" in df.columns:
        df["Education_Level"].replace({"Highschool": "High School"}, inplace=True)

    # County normalization
    if "County" in df.columns:
        df["County"].replace({"Trans Nzoia": "Trans-Nzoia"}, inplace=True)

    _analysis_cache = (df, None)
    return _analysis_cache


def _generate_analysis_charts(df):
    """Generate interactive charts as HTML snippets (Plotly)."""
    charts = {
        "graduates_branch": None,
        "transition_bar": None,
        "transition_pie": None,
        "course_distribution": None,
        "gender_branch": None,
        "age_group_dist": None,
        "age_group_year": None,
        "age_group_gender_year": None,
    }

    # 1) Graduates per branch by year
    if {"Year of Class", "Branch"}.issubset(df.columns):
        grouped = df.groupby(["Year of Class", "Branch"]).size().reset_index(name="count")
        fig = px.bar(
            grouped,
            x="Year of Class",
            y="count",
            color="Branch",
            barmode="group",
            title="Number of Graduates per Branch by Year",
        )
        fig.update_layout(
            autosize=True,
            height=520,
            margin=dict(l=30, r=30, t=60, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            title=dict(
                text="<b>Number of Graduates per Branch by Year</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color="#111827", family="Inter, Arial, sans-serif"),
            ),
        )
        fig.update_xaxes(tickangle=-45)
        charts["graduates_branch"] = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # 2) Transition rate by course (bar) and overall pie
    if {"Course", "Transitioned_to_Caregiving"}.issubset(df.columns):
        transition_data = df.groupby(["Course", "Transitioned_to_Caregiving"]).size().reset_index(name="count")
        fig_bar = px.bar(
            transition_data,
            x="Course",
            y="count",
            color="Transitioned_to_Caregiving",
            barmode="group",
            title="Transition rate by Course",
        )
        fig_bar.update_layout(
            margin=dict(l=20, r=20, t=40, b=40),
            title=dict(
                text="<b>Transition rate by Course</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color="#111827", family="Inter, Arial, sans-serif"),
            ),
        )
        charts["transition_bar"] = fig_bar.to_html(full_html=False, include_plotlyjs="cdn")

        transition_overall = df["Transitioned_to_Caregiving"].value_counts()
        fig_pie = px.pie(
            names=transition_overall.index,
            values=transition_overall.values,
            title="Overall Transition to Caregiving",
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(
            margin=dict(l=20, r=20, t=40, b=40),
            title=dict(
                text="<b>Overall Transition to Caregiving</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color="#111827", family="Inter, Arial, sans-serif"),
            ),
        )
        charts["transition_pie"] = fig_pie.to_html(full_html=False, include_plotlyjs="cdn")

    # 3) Course distribution
    if "Course" in df.columns:
        course_counts = df["Course"].value_counts().reset_index()
        course_counts.columns = ["Course", "count"]
        fig = px.bar(
            course_counts,
            x="Course",
            y="count",
            color="Course",
            title="Course Distribution",
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=40),
            showlegend=False,
            title=dict(
                text="<b>Course Distribution</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color="#111827", family="Inter, Arial, sans-serif"),
            ),
        )
        charts["course_distribution"] = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # 4) Gender distribution by branch
    if {"Branch", "Gender"}.issubset(df.columns):
        gender_branch = df.groupby(["Branch", "Gender"]).size().reset_index(name="count")
        fig = px.bar(
            gender_branch,
            x="Branch",
            y="count",
            color="Gender",
            barmode="group",
            text_auto=True,
            title="Gender Distribution by Branch",
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            title=dict(
                text="<b>Gender Distribution by Branch</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color="#111827", family="Inter, Arial, sans-serif"),
            ),
        )
        fig.update_xaxes(tickangle=-45)
        charts["gender_branch"] = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # 5) Age group distribution and by class year
    if "Age" in df.columns:
        bins = [17, 24, 34, 44, 54, 65]
        labels = ['18–24', '25–34', '35–44', '45–54', '55–60']
        df_age = df.copy()
        df_age["Age Group"] = pd.cut(df_age["Age"], bins=bins, labels=labels, right=True, include_lowest=True)

        age_counts = df_age["Age Group"].value_counts().reindex(labels).reset_index()
        age_counts.columns = ["Age Group", "count"]
        fig_age = px.bar(
            age_counts,
            x="Age Group",
            y="count",
            color="Age Group",
            text_auto=True,
            title="Distribution of Age Groups",
        )
        fig_age.update_layout(
            margin=dict(l=20, r=20, t=40, b=40),
            showlegend=False,
            title=dict(
                text="<b>Distribution of Age Groups</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=18, color="#111827", family="Inter, Arial, sans-serif"),
            ),
        )
        charts["age_group_dist"] = fig_age.to_html(full_html=False, include_plotlyjs="cdn")

        if "Year of Class" in df_age.columns:
            age_year = df_age.groupby(["Year of Class", "Age Group"]).size().reset_index(name="count")
            fig_age_year = px.bar(
                age_year,
                x="Year of Class",
                y="count",
                color="Age Group",
                barmode="group",
                title="Age Group Distribution by Class Start Year",
            )
            fig_age_year.update_layout(
                margin=dict(l=20, r=20, t=40, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                title=dict(
                    text="<b>Age Group Distribution by Class Start Year</b>",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=18, color="#111827", family="Inter, Arial, sans-serif"),
                ),
            )
            fig_age_year.update_xaxes(tickangle=-45)
            charts["age_group_year"] = fig_age_year.to_html(full_html=False, include_plotlyjs="cdn")

        # Age group by gender with year table
        if {"Gender", "Year of Class"}.issubset(df_age.columns):
            # Bar
            order_labels = labels
            age_gender = df_age.groupby(["Age Group", "Gender"]).size().reset_index(name="count")
            fig_bar = px.bar(
                age_gender,
                x="Age Group",
                y="count",
                color="Gender",
                category_orders={"Age Group": order_labels},
                barmode="group",
                title="Distribution of Age Groups by Gender",
                text_auto=True,
            )
            fig_bar.update_layout(
                margin=dict(l=20, r=20, t=50, b=120),
                title=dict(
                    text="<b>Distribution of Age Groups by Gender</b>",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=18, color="#111827", family="Inter, Arial, sans-serif"),
                ),
            )
            fig_bar.update_yaxes(title="Number of Students")

            # Table data
            table_data = df_age.groupby(["Year of Class", "Age Group", "Gender"]).size().unstack(fill_value=0)
            table_df = table_data.reset_index()
            table_df["Age-Year"] = table_df["Year of Class"].astype(str) + " | " + table_df["Age Group"].astype(str)

            female_col = table_df.columns[table_df.columns.str.lower() == "female"]
            male_col = table_df.columns[table_df.columns.str.lower() == "male"]
            female_name = female_col[0] if len(female_col) else None
            male_name = male_col[0] if len(male_col) else None

            table_display = table_df[["Age-Year"] + ([female_name] if female_name else []) + ([male_name] if male_name else [])]
            table_display = table_display.rename(columns={female_name: "Female", male_name: "Male"})

            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig_combined = make_subplots(
                rows=2, cols=1,
                specs=[[{"type": "xy"}], [{"type": "table"}]],
                row_heights=[0.6, 0.4],
                vertical_spacing=0.08,
            )

            for trace in fig_bar.data:
                fig_combined.add_trace(trace, row=1, col=1)

            fig_combined.update_xaxes(title_text="Age Group", row=1, col=1, tickangle=0)
            fig_combined.update_yaxes(title_text="Number of Students", row=1, col=1)

            fig_combined.add_trace(
                go.Table(
                    header=dict(values=list(table_display.columns), fill_color="#667eea", font=dict(color="white", size=12), align="center"),
                    cells=dict(values=[table_display[col] for col in table_display.columns], align="center"),
                ),
                row=2, col=1
            )

            fig_combined.update_layout(
                height=850,
                margin=dict(l=20, r=20, t=60, b=20),
                title=dict(
                    text="<b>Distribution of Age Groups by Gender with Year of Class</b>",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=18, color="#111827", family="Inter, Arial, sans-serif"),
                ),
            )

            charts["age_group_gender_year"] = fig_combined.to_html(full_html=False, include_plotlyjs="cdn")

    return charts


def home(request):
    """Home page view"""
    return render(request, "home.html")


def about(request):
    """About page view"""
    return render(request, "about.html")


def predict_form(request):
    """Prediction form view"""
    context = {
        'features_info': features_info,
        'categorical_values': categorical_values,
        'county_subcounty_map': county_subcounty_map
    }
    return render(request, "predict.html", context)


def get_subcounties(request, county):
    """Get subcounties for a given county"""
    subcounties = county_subcounty_map.get(county, [])
    return JsonResponse(subcounties, safe=False)


@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    """Chat endpoint for AI assistant"""
    import json
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').lower()
    except:
        user_message = ''
    
    # Simple AI responses based on keywords
    if 'predict' in user_message or 'prediction' in user_message:
        response = "To make a prediction, fill out all the form fields including your course, grades, and location details. The system will analyze your data and provide a care transition likelihood."
    elif 'score' in user_message or 'grade' in user_message:
        response = "Scores below 70 in any assessment may result in lower transition probability. A 'Fail' grade automatically results in 'Likely not to transition'."
    elif 'duration' in user_message or 'months' in user_message:
        response = "Minimum training duration: Childcare requires 2+ months, Eldercare requires 3+ months. Not meeting these requirements results in 'Likely not to transition'."
    elif 'county' in user_message or 'location' in user_message:
        response = "Select your county first, then the subcounty dropdown will automatically populate with relevant options for your area."
    elif 'course' in user_message:
        response = "Choose between Childcare or Eldercare courses. Each has different minimum duration requirements for successful transition."
    elif 'help' in user_message or 'how' in user_message:
        response = "I can help you understand the prediction form, scoring requirements, duration rules, and location selection. What specific question do you have?"
    else:
        response = "I'm here to help with the Care Transition Prediction system. Ask me about form fields, scoring requirements, duration rules, or how predictions work!"
    
    return JsonResponse({'response': response})


@require_http_methods(["POST"])
def result(request):
    """Prediction result view"""
    input_data = {}
    for feat in top_features:
        val = request.POST.get(feat)
        if feat in form_categorical_features:
            input_data[feat] = val if val else ""
        else:
            input_data[feat] = float(val) if val else 0.0
    
    # Business Rules - Check BEFORE encoding (use original string values)
    score_features = ['Final Score', 'Practical-Icare', 'Average Attendance', 
                     'Average Practicals', 'Theory Exam - Icare', 'Hospital Internship Score', 
                     'Average Cats']
    
    # Check if any score is below 70
    low_scores = []
    for feature in score_features:
        if feature in input_data:
            score = input_data[feature]
            if isinstance(score, (int, float)) and score < 70:
                low_scores.append(feature)
    
    # Check if Final Grade is 'Fail'
    final_grade_fail = input_data.get('Final Grade') == 'Fail'
    
    # Check course duration requirements
    course = input_data.get('Course', '')
    duration = input_data.get('Course Duration Months', 0)
    duration_fail = False
    duration_reason = ""
    
    if course == 'Childcare' and float(duration) < 2:
        duration_fail = True
        duration_reason = "For Childcare, the minimum training duration should be 2 months which was not met"
    elif course == 'Eldercare' and float(duration) < 3:
        duration_fail = True
        duration_reason = "The minimum training time for Eldercare course is 3 months, which was not met with the caregiver"
    
    # Initialize reasons list
    failure_reasons = []
    
    # Apply business rules in order of severity - DURATION IS HIGHEST PRIORITY
    
    if duration_fail:
        # Duration requirement not met - OVERRIDES EVERYTHING (even good grades)
        will_transition = False
        probability = 0.05
        failure_reasons.append(duration_reason)
    elif final_grade_fail:
        # Final grade failure - cannot transition
        will_transition = False
        probability = 0.10
        failure_reasons.append("Final Grade is 'Fail'")
    elif low_scores:
        # Low scores but no fail - can transition with low probability
        will_transition = True
        probability = 0.25  # Low but possible transition
        failure_reasons.append(f"Low performance in: {', '.join(low_scores)} (below 70)")
    else:
        # All business rules pass - use ML model
        try:
            input_df = pd.DataFrame([input_data], columns=top_features)
            
            # Encode categorical features with error handling for unseen labels
            for col in form_categorical_features:
                if col in input_df.columns and col in cat_to_encoder_idx:
                    idx = cat_to_encoder_idx[col]
                    encoder = encoders[idx]
                    
                    # Handle unseen labels
                    original_value = str(input_df[col].iloc[0]).strip()
                    if original_value not in encoder.classes_:
                        # Find closest match or use most common class
                        if col == 'Sub_County':
                            # For subcounty, try to find a match in the same county
                            county_value = input_data.get('County', '')
                            if county_value in county_subcounty_map:
                                subcounties = county_subcounty_map[county_value]
                                # Find closest match or use first subcounty
                                closest_match = None
                                for sc in subcounties:
                                    if sc in encoder.classes_:
                                        closest_match = sc
                                        break
                                if closest_match:
                                    input_df[col] = closest_match
                                else:
                                    # Use most common subcounty from encoder
                                    input_df[col] = encoder.classes_[0]
                            else:
                                input_df[col] = encoder.classes_[0]
                        else:
                            # For other categories, use most common class
                            input_df[col] = encoder.classes_[0]
                    
                    # Now encode the (possibly corrected) value
                    input_df[col] = encoder.transform(input_df[col].astype(str))
            
            # Reorder columns to match scaler's expected order
            scaler_feature_order = scaler.feature_names_in_.tolist()
            input_df_scaled = input_df[scaler_feature_order]
            
            # Scale all features
            input_df_scaled = pd.DataFrame(scaler.transform(input_df_scaled), columns=scaler_feature_order)
            
            # Reorder back to top_features order for model
            input_df = input_df_scaled[top_features]
            
            # Use ML model prediction
            pred_prob = model.predict_proba(input_df)[:, 1][0]
            pred = model.predict(input_df)[0]
            probability = round(pred_prob, 4)
            will_transition = pred == 1
            
        except Exception as e:
            # If ML model fails, provide conservative prediction
            will_transition = True
            probability = 0.50  # Neutral probability
            failure_reasons.append(f"Model prediction unavailable, using default estimate")
    
    context = {
        'will_transition': will_transition,
        'probability': probability,
        'probability_percentage': round(probability * 100, 2),
        'failure_reasons': failure_reasons,
        'low_scores': low_scores,
        'final_grade_fail': final_grade_fail,
        'duration_fail': duration_fail
    }
    
    return render(request, "result.html", context)


def analysis(request):
    """Data analysis view with cleaning pipeline and summary metrics."""
    df, error = _load_and_clean_dataset()
    if error:
        return render(request, "analysis.html", {"error": error["error"]})

    charts = _generate_analysis_charts(df)

    row_count = len(df)
    col_count = len(df.columns)

    def value_counts_dict(col):
        return df[col].value_counts().to_dict() if col in df.columns else {}

    grade_counts = value_counts_dict("Final Grade")
    course_counts = value_counts_dict("Course")
    transition_counts = value_counts_dict("Transitioned_to_Caregiving")

    context = {
        "row_count": row_count,
        "col_count": col_count,
        "grade_counts": grade_counts,
        "course_counts": course_counts,
        "transition_counts": transition_counts,
        "charts": charts,
    }
    return render(request, "analysis.html", context)