import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup and model loading
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ABC Ltd Employee Retention Assistant", page_icon="👥", layout="wide")


@st.cache_resource
def load_models():
    models = pickle.load(open('hr_model.sav', 'rb'))   # one file holds both models
    return models['logistic'], models['linear']


logit_model, linear_model = load_models()

# ---------------------------------------------------------------------------
# Settings (taken from the Colab notebook)
# ---------------------------------------------------------------------------
LOGIT_FEATURES = list(logit_model.feature_names_in_)
LINREG_FEATURES = list(linear_model.feature_names_in_)

# Test-set results from the notebook
RECALL = 0.68
R2, MAE = 0.94, 890
LOW_CUT, HIGH_CUT = 0.30, 0.60   # risk bands

JOB_ROLES = ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager',
             'Manufacturing Director', 'Research Director', 'Research Scientist',
             'Sales Executive', 'Sales Representative']
# Job levels that actually exist for each role in ABC Ltd's data
ROLE_LEVELS = {'Healthcare Representative': (2, 4), 'Human Resources': (1, 3),
               'Laboratory Technician': (1, 3), 'Manager': (3, 5), 'Manufacturing Director': (2, 4),
               'Research Director': (3, 5), 'Research Scientist': (1, 3),
               'Sales Executive': (2, 4), 'Sales Representative': (1, 2)}

SURVEY = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
WORK_LIFE = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
EDUCATION = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
PERFORMANCE = {3: 'Excellent', 4: 'Outstanding'}
TRAVEL = {'Non-Travel': 'Does not travel', 'Travel_Rarely': 'Travels rarely',
          'Travel_Frequently': 'Travels frequently'}

FRIENDLY = {
    'Age': 'Age', 'MaritalStatus': 'Marital status', 'DistanceFromHome': 'Distance from home',
    'NumCompaniesWorked': 'Number of previous employers', 'TotalWorkingYears': 'Total work experience',
    'JobRole': 'Job role', 'JobLevel': 'Job level', 'MonthlyIncome': 'Monthly income',
    'StockOptionLevel': 'Stock option level', 'OverTime': 'Overtime', 'BusinessTravel': 'Business travel',
    'YearsAtCompany': 'Years at ABC Ltd', 'YearsWithCurrManager': 'Years with current manager',
    'YearsSinceLastPromotion': 'Years since last promotion',
    'TrainingTimesLastYear': 'Trainings attended last year', 'JobSatisfaction': 'Job satisfaction',
    'EnvironmentSatisfaction': 'Satisfaction with work environment',
    'RelationshipSatisfaction': 'Satisfaction with colleagues', 'WorkLifeBalance': 'Work-life balance',
    'JobInvolvement': 'Job involvement',
}

ORANGE, BLUE = "#eb6834", "#2a78d6"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def risk_score(employee):
    return float(logit_model.predict_proba(employee[LOGIT_FEATURES])[0, 1])


def fair_salary(employee):
    return float(linear_model.predict(employee[LINREG_FEATURES])[0])


def risk_band(score):
    if score >= HIGH_CUT:
        return "High"
    if score >= LOW_CUT:
        return "Medium"
    return "Low"


def show_value(feature, value):
    """Turn a raw value into words a manager understands."""
    if feature in ('JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'JobInvolvement'):
        return SURVEY[value]
    if feature == 'WorkLifeBalance':
        return WORK_LIFE[value]
    if feature == 'BusinessTravel':
        return TRAVEL[value]
    if feature == 'OverTime':
        return 'works overtime' if value == 'Yes' else 'no overtime'
    if feature == 'MonthlyIncome':
        return f"{value:,.0f} per month"
    if feature in ('TotalWorkingYears', 'YearsAtCompany', 'YearsWithCurrManager', 'YearsSinceLastPromotion'):
        return f"{value} years"
    if feature == 'DistanceFromHome':
        return f"{value} km"
    return str(value)


def explain(employee):
    """How much each factor pushes this employee's risk up or down,
    compared with an average ABC Ltd employee (coefficient x difference from average)."""
    prep = logit_model.named_steps['prep']
    coef = logit_model.named_steps['model'].coef_[0]
    x = prep.transform(employee[LOGIT_FEATURES])
    x = np.asarray(x.toarray() if hasattr(x, 'toarray') else x).ravel()
    contrib = coef * (x - logit_model.average_employee_)

    names = [n.split('__', 1)[1] for n in prep.get_feature_names_out()]
    totals = {}
    for name, c in zip(names, contrib):
        feature = next(f for f in LOGIT_FEATURES if name == f or name.startswith(f + '_'))
        totals[feature] = totals.get(feature, 0) + c

    rows = [{'Factor': FRIENDLY[f], 'Value': show_value(f, employee[f].iloc[0]), 'Effect': v}
            for f, v in totals.items() if abs(v) >= 0.05]
    return pd.DataFrame(rows, columns=['Factor', 'Value', 'Effect']).sort_values('Effect', ascending=False)


def effect_chart(df):
    df = df.copy()
    df['Label'] = df['Factor'] + ' (' + df['Value'] + ')'
    df['Direction'] = np.where(df['Effect'] > 0, 'Raises risk', 'Lowers risk')
    return (alt.Chart(df).mark_bar(cornerRadius=3, height=16)
            .encode(x=alt.X('Effect:Q', title='← lowers risk          raises risk →'),
                    y=alt.Y('Label:N', sort='-x', title=None),
                    color=alt.Color('Direction:N', scale=alt.Scale(domain=['Raises risk', 'Lowers risk'],
                                                                   range=[ORANGE, BLUE]),
                                    legend=alt.Legend(orient='top', title=None)),
                    tooltip=['Factor', 'Value', alt.Tooltip('Effect:Q', format='+.2f')])
            .properties(height=max(200, 26 * len(df))))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("👥 ABC Ltd")
    st.subheader("Employee Retention Assistant")
    st.write("A decision-support tool for managers. It uses ABC Ltd's past HR data "
             "(1,470 employees) to estimate:")
    st.markdown("- **Attrition risk**: how likely an employee is to leave\n"
                "- **Fair pay**: what ABC Ltd typically pays for a similar role and experience")
    st.divider()
    st.caption("**How accurate is it?**")
    st.caption(f"Catches about **{RECALL:.0%}** of employees who later leave. "
               f"Of the employees it flags, about 1 in 3 actually leave; the rest are false alarms.")
    st.caption(f"Salary benchmark is usually within **±{MAE:,}** per month of actual pay.")
    st.divider()
    st.caption("⚠️ This tool supports, not replaces, your judgement. Use it to start a conversation, "
               "never as the only basis for a decision about a person.")

st.title("Employee Retention Assistant")
tab1, tab2 = st.tabs(["🔍 Check an employee", "💰 Salary benchmark"])

# ---------------------------------------------------------------------------
# TAB 1: Attrition risk + fair pay for one employee
# ---------------------------------------------------------------------------
with tab1:
    st.write("Fill in what you know about the employee and press **Assess employee**. "
             "Defaults are set to a typical ABC Ltd employee.")

    with st.form('employee_form'):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**About the employee**")
            age = st.number_input('Age', 18, 60, 36)
            marital = st.selectbox('Marital status', ['Single', 'Married', 'Divorced'], index=1)
            distance = st.number_input('Distance from home (km)', 1, 29, 7)
            num_companies = st.number_input('Number of previous employers', 0, 9, 2)
            total_years = st.number_input('Total work experience (years)', 0, 40, 10)
            education = st.selectbox('Education', list(EDUCATION), index=2, format_func=EDUCATION.get)
        with c2:
            st.markdown("**Job & pay**")
            role = st.selectbox('Job role', JOB_ROLES, index=7)
            level = st.selectbox('Job level (1 = entry, 5 = senior)', [1, 2, 3, 4, 5], index=1)
            income = st.number_input('Current monthly income', 1000, 20000, 5500, step=100)
            stock = st.selectbox('Stock option level (0 = none)', [0, 1, 2, 3], index=1)
            overtime = st.radio('Works overtime?', ['No', 'Yes'], horizontal=True)
            travel = st.selectbox('Business travel', list(TRAVEL), index=1, format_func=TRAVEL.get)
            performance = st.selectbox('Last performance rating', list(PERFORMANCE), format_func=PERFORMANCE.get)
        with c3:
            st.markdown("**Time at ABC Ltd**")
            years_company = st.number_input('Years at ABC Ltd', 0, 40, 5)
            years_manager = st.number_input('Years with current manager', 0, 17, 3)
            years_promo = st.number_input('Years since last promotion', 0, 15, 1)
            training = st.number_input('Trainings attended last year', 0, 6, 3)

        st.markdown("**How the employee feels** (from surveys or your own impression)")
        s1, s2, s3, s4, s5 = st.columns(5)
        job_sat = s1.select_slider('Job satisfaction', list(SURVEY), 3, format_func=SURVEY.get)
        env_sat = s2.select_slider('Work environment', list(SURVEY), 3, format_func=SURVEY.get)
        rel_sat = s3.select_slider('Relationships with colleagues', list(SURVEY), 3, format_func=SURVEY.get)
        wlb = s4.select_slider('Work-life balance', list(WORK_LIFE), 3, format_func=WORK_LIFE.get)
        involvement = s5.select_slider('Job involvement', list(SURVEY), 3, format_func=SURVEY.get)

        submitted = st.form_submit_button('Assess employee', type='primary', width='stretch')

    if submitted:
        st.session_state['employee'] = pd.DataFrame([{
            'Age': age, 'MaritalStatus': marital, 'DistanceFromHome': distance,
            'NumCompaniesWorked': num_companies, 'TotalWorkingYears': total_years,
            'JobRole': role, 'JobLevel': level, 'MonthlyIncome': income, 'StockOptionLevel': stock,
            'OverTime': overtime, 'BusinessTravel': travel, 'YearsAtCompany': years_company,
            'YearsWithCurrManager': years_manager, 'YearsSinceLastPromotion': years_promo,
            'TrainingTimesLastYear': training, 'JobSatisfaction': job_sat,
            'EnvironmentSatisfaction': env_sat, 'RelationshipSatisfaction': rel_sat,
            'WorkLifeBalance': wlb, 'JobInvolvement': involvement,
            'Education': education, 'PerformanceRating': performance,
        }])

    if 'employee' in st.session_state:
        emp = st.session_state['employee']
        e = emp.iloc[0]

        # Warnings when the inputs are unusual for ABC Ltd (prediction less reliable)
        lo, hi = ROLE_LEVELS[e['JobRole']]
        if not lo <= e['JobLevel'] <= hi:
            st.warning(f"At ABC Ltd, {e['JobRole']}s are at job level {lo}–{hi}. Level {e['JobLevel']} "
                       "has not occurred before, so treat these results with extra caution.")
        if e['YearsAtCompany'] > e['TotalWorkingYears']:
            st.warning("Years at ABC Ltd is more than total work experience. Please check the inputs.")

        risk = risk_score(emp)
        band = risk_band(risk)
        fair = fair_salary(emp)
        pay_gap = (e['MonthlyIncome'] - fair) / fair

        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Attrition risk")
            st.metric("Risk score", f"{risk:.0%}")
            st.progress(risk)
            if band == "High":
                st.error("🔴 **High risk.** About half of past ABC Ltd employees with a score this high left.")
            elif band == "Medium":
                st.warning("🟠 **Medium risk.** About 1 in 8 past employees with a score in this range left.")
            else:
                st.success("🟢 **Low risk.** Only about 3 in 100 past employees with a score this low left.")
        with r2:
            st.subheader("Pay check")
            st.metric("Fair monthly income (benchmark)", f"{fair:,.0f}",
                      delta=f"{pay_gap:+.0%} current pay vs benchmark", delta_color='normal')
            st.caption(f"Typical range for this profile: {fair - MAE:,.0f} to {fair + MAE:,.0f}")
            if pay_gap < -0.15:
                st.warning("Paid noticeably **below** what ABC Ltd pays similar employees.")
            elif pay_gap > 0.15:
                st.info("Paid noticeably **above** what ABC Ltd pays similar employees.")
            else:
                st.success("Pay is in line with similar employees.")

        # Why?
        st.subheader("Why this score?")
        st.caption("Each bar shows how much a factor pushes this person's risk up or down "
                   "compared with an average ABC Ltd employee.")
        reasons = explain(emp)
        st.altair_chart(effect_chart(reasons), width='stretch')

        # What-if scenarios a manager can act on
        scenarios = []
        if e['OverTime'] == 'Yes':
            scenarios.append(('Remove overtime', {'OverTime': 'No'}))
        if e['BusinessTravel'] == 'Travel_Frequently':
            scenarios.append(('Reduce travel to "rarely"', {'BusinessTravel': 'Travel_Rarely'}))
        if e['YearsSinceLastPromotion'] >= 2:
            scenarios.append(('Promote now (reset years since promotion)', {'YearsSinceLastPromotion': 0}))
        if pay_gap < -0.10:
            scenarios.append((f'Raise pay to benchmark ({fair:,.0f})', {'MonthlyIncome': round(fair)}))
        if e['StockOptionLevel'] == 0:
            scenarios.append(('Grant stock options (level 1)', {'StockOptionLevel': 1}))
        for col, label in [('JobSatisfaction', 'job satisfaction'),
                           ('EnvironmentSatisfaction', 'work environment satisfaction'),
                           ('WorkLifeBalance', 'work-life balance')]:
            if e[col] <= 2:
                scenarios.append((f'Improve {label} to the next level', {col: e[col] + 1}))

        st.subheader("What could reduce the risk?")
        if scenarios:
            rows = []
            for name, change in scenarios:
                new = emp.copy()
                for k, v in change.items():
                    new[k] = v
                new_risk = risk_score(new)
                points = round((new_risk - risk) * 100)
                rows.append({'If the manager…': name, 'New risk score': f"{new_risk:.0%}",
                             'Change': f"{points:+d} points" if points else "almost no change"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            st.caption("Estimates from past patterns at ABC Ltd, not guarantees. "
                       "Each row changes one thing at a time.")
        else:
            st.write("No obvious levers found: no overtime, frequent travel, long wait for promotion, "
                     "low pay or low satisfaction scores.")

# ---------------------------------------------------------------------------
# TAB 2: Salary benchmark (linear regression)
# ---------------------------------------------------------------------------
with tab2:
    st.write("Get a fair-pay benchmark for a **new hire, promotion or salary review**, "
             "based on what ABC Ltd pays employees with a similar profile.")
    with st.form('salary_form'):
        a, b, c = st.columns(3)
        s_role = a.selectbox('Job role', JOB_ROLES, index=7, key='s_role')
        s_level = a.selectbox('Job level (1 = entry, 5 = senior)', [1, 2, 3, 4, 5], index=1, key='s_level')
        s_total = b.number_input('Total work experience (years)', 0, 40, 8, key='s_total')
        s_company = b.number_input('Years at ABC Ltd (0 for a new hire)', 0, 40, 0, key='s_company')
        s_age = b.number_input('Age', 18, 60, 32, key='s_age')
        s_edu = c.selectbox('Education', list(EDUCATION), index=2, format_func=EDUCATION.get, key='s_edu')
        s_perf = c.selectbox('Performance rating', list(PERFORMANCE), format_func=PERFORMANCE.get, key='s_perf')
        s_prev = c.number_input('Number of previous employers', 0, 9, 2, key='s_prev')
        s_go = st.form_submit_button('Get benchmark', type='primary', width='stretch')

    if s_go:
        cand = pd.DataFrame([{'JobLevel': s_level, 'JobRole': s_role, 'TotalWorkingYears': s_total,
                              'YearsAtCompany': s_company, 'Age': s_age, 'Education': s_edu,
                              'PerformanceRating': s_perf, 'NumCompaniesWorked': s_prev}])
        lo, hi = ROLE_LEVELS[s_role]
        if not lo <= s_level <= hi:
            st.warning(f"At ABC Ltd, {s_role}s are at job level {lo}–{hi}. "
                       "This combination has not occurred before, so the benchmark is less reliable.")
        pay = fair_salary(cand)
        m1, m2, m3 = st.columns(3)
        m1.metric("Benchmark monthly income", f"{pay:,.0f}")
        m2.metric("Lower end", f"{pay - MAE:,.0f}")
        m3.metric("Upper end", f"{pay + MAE:,.0f}")
        st.caption(f"The model explains {R2:.0%} of the differences in pay at ABC Ltd. Job level and role "
                   "matter most; experience adds a little; education and performance rating barely change pay.")
