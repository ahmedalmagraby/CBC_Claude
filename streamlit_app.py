import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import statsmodels.api as sm
from statsmodels.formula.api import mnlogit
from scipy.special import softmax
from itertools import product

# Page configuration
st.set_page_config(page_title="CBC Conjoint Analysis", layout="wide", page_icon="📊")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'individual_utilities' not in st.session_state:
    st.session_state.individual_utilities = None

# Define attribute structure (customize based on your study)
DEFAULT_ATTRIBUTES = {
    'Brand': ['Ciao', 'Amanda', 'Bella', 'Carmex'],
    'Price': ['$5', '$8', '$12', '$15'],
    'Hero_Claim': ['Long Lasting', 'Moisturizing', 'Natural', 'Tinted'],
    'Ingredients': ['Hyaluronic Acid', 'Vitamin E', 'Shea Butter', 'Beeswax'],
    'SPF': ['No SPF', 'SPF 15', 'SPF 30', 'SPF 50']
}

def generate_simulated_data(n_respondents=100, n_questions=12, attributes=DEFAULT_ATTRIBUTES):
    """Generate simulated conjoint data for testing"""
    
    np.random.seed(42)
    data = []
    
    # Generate true part-worths for simulation
    true_utilities = {}
    for attr, levels in attributes.items():
        # Random utilities that sum to zero (effects coding)
        utils = np.random.randn(len(levels))
        utils = utils - utils.mean()
        true_utilities[attr] = dict(zip(levels, utils))
    
    for resp_id in range(1, n_respondents + 1):
        # Add some individual variation
        individual_utilities = {}
        for attr, level_utils in true_utilities.items():
            individual_utilities[attr] = {
                level: util + np.random.randn() * 0.3 
                for level, util in level_utils.items()
            }
        
        for question in range(1, n_questions + 1):
            # Generate 3 random profiles
            profiles = []
            for profile_num in range(1, 4):
                profile = {}
                for attr, levels in attributes.items():
                    profile[attr] = np.random.choice(levels)
                profiles.append(profile)
            
            # Calculate utility for each profile
            profile_utilities = []
            for profile in profiles:
                utility = sum(
                    individual_utilities[attr][profile[attr]] 
                    for attr in attributes.keys()
                )
                profile_utilities.append(utility)
            
            # Add random error and select based on highest utility
            profile_utilities = np.array(profile_utilities) + np.random.gumbel(0, 1, 3)
            selected = np.argmax(profile_utilities) + 1
            
            # Create row
            row = {
                'Respondent_ID': resp_id,
                'Question': question,
                'Selected_Profile': selected
            }
            
            # Add profile data
            for profile_num, profile in enumerate(profiles, 1):
                for attr, level in profile.items():
                    row[f'Profile_{profile_num}_{attr}'] = level
            
            data.append(row)
    
    return pd.DataFrame(data)

def create_example_template():
    """Create example Excel template"""
    example_data = generate_simulated_data(n_respondents=5, n_questions=3)
    return example_data

def validate_data(df):
    """Validate uploaded data format"""
    required_cols = ['Respondent_ID', 'Question', 'Selected_Profile']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Check for profile columns
    profile_cols = [col for col in df.columns if col.startswith('Profile_')]
    if len(profile_cols) == 0:
        return False, "No profile columns found (should start with 'Profile_')"
    
    return True, "Data format validated successfully!"

def reshape_data_to_long(df, attributes):
    """Reshape wide format data to long format for analysis"""
    
    long_data = []
    
    for _, row in df.iterrows():
        respondent_id = row['Respondent_ID']
        question = row['Question']
        selected_profile = row['Selected_Profile']
        
        # Extract data for each of the 3 profiles
        for profile_num in range(1, 4):
            profile_data = {
                'Respondent_ID': respondent_id,
                'Question': question,
                'Profile': profile_num,
                'Choice': 1 if profile_num == selected_profile else 0
            }
            
            # Extract attribute levels for this profile
            for attr in attributes.keys():
                col_name = f'Profile_{profile_num}_{attr}'
                if col_name in df.columns:
                    profile_data[attr] = row[col_name]
            
            long_data.append(profile_data)
    
    long_df = pd.DataFrame(long_data)
    
    # Create choice set identifier
    long_df['Choice_Set'] = long_df['Respondent_ID'].astype(str) + '_' + long_df['Question'].astype(str)
    
    return long_df

def effects_code_attributes(df, attributes):
    """Apply effects coding to attributes (sum to zero constraint)"""
    
    df_coded = df.copy()
    coding_map = {}
    
    for attr, levels in attributes.items():
        if attr in df_coded.columns:
            # Effects coding: last level is reference (-1 for all dummies)
            for i, level in enumerate(levels[:-1]):  # All except last
                col_name = f'{attr}_{level}'
                df_coded[col_name] = (df_coded[attr] == level).astype(int)
                
                # If this is not the last dummy, we're done
                # If it's any level except the reference, mark it
            
            # For reference level, all dummies are -1
            reference_level = levels[-1]
            is_reference = df_coded[attr] == reference_level
            for i, level in enumerate(levels[:-1]):
                col_name = f'{attr}_{level}'
                df_coded.loc[is_reference, col_name] = -1
            
            coding_map[attr] = {
                'levels': levels,
                'reference': reference_level,
                'coded_columns': [f'{attr}_{level}' for level in levels[:-1]]
            }
    
    return df_coded, coding_map

def fit_conditional_logit(df_long, attributes):
    """Fit conditional logit model using statsmodels"""
    
    # Effects code the data
    df_coded, coding_map = effects_code_attributes(df_long, attributes)
    
    # Get all coded attribute columns
    coded_cols = []
    for attr_info in coding_map.values():
        coded_cols.extend(attr_info['coded_columns'])
    
    # Prepare data for conditional logit
    # We need to use the mnlogit with choice set structure
    X = df_coded[coded_cols].values
    y = df_coded['Choice'].values
    groups = df_coded['Choice_Set'].values
    
    # Fit using conditional logit approach
    # Create choice set dummies
    choice_sets = pd.get_dummies(df_coded['Choice_Set'], drop_first=False)
    
    # For conditional logit, we use a different approach
    # We'll use a simplified estimation with fixed effects by choice set
    
    # Alternative: Use direct maximum likelihood
    from scipy.optimize import minimize
    
    def negative_log_likelihood(params):
        """Negative log-likelihood for conditional logit"""
        utilities = X @ params
        
        # Group by choice set and calculate probabilities
        nll = 0
        for choice_set in df_coded['Choice_Set'].unique():
            mask = df_coded['Choice_Set'] == choice_set
            set_utilities = utilities[mask]
            set_choices = y[mask]
            
            # Softmax probabilities
            probs = softmax(set_utilities)
            
            # Log likelihood of chosen alternative
            chosen_idx = np.where(set_choices == 1)[0][0]
            nll -= np.log(probs[chosen_idx] + 1e-10)
        
        return nll
    
    # Initial parameters
    initial_params = np.zeros(len(coded_cols))
    
    # Optimize
    result = minimize(negative_log_likelihood, initial_params, method='BFGS')
    
    # Extract coefficients
    coefficients = dict(zip(coded_cols, result.x))
    
    # Calculate part-worth utilities for all levels (including reference)
    utilities = {}
    for attr, attr_info in coding_map.items():
        utilities[attr] = {}
        
        # Get utilities for coded levels
        for level in attr_info['levels'][:-1]:
            col_name = f'{attr}_{level}'
            utilities[attr][level] = coefficients[col_name]
        
        # Reference level utility (negative sum of others for effects coding)
        reference_util = -sum(utilities[attr].values())
        utilities[attr][attr_info['reference']] = reference_util
    
    return {
        'coefficients': coefficients,
        'utilities': utilities,
        'coding_map': coding_map,
        'optimization_result': result,
        'log_likelihood': -result.fun
    }

def calculate_attribute_importance(utilities):
    """Calculate relative importance of each attribute"""
    
    importance = {}
    ranges = {}
    
    # Calculate range for each attribute
    for attr, levels in utilities.items():
        level_utils = list(levels.values())
        attr_range = max(level_utils) - min(level_utils)
        ranges[attr] = attr_range
    
    # Calculate relative importance
    total_range = sum(ranges.values())
    for attr, attr_range in ranges.items():
        importance[attr] = (attr_range / total_range) * 100
    
    return importance, ranges

def calculate_individual_utilities(df_long, attributes, aggregate_utilities):
    """Calculate individual-level utilities using empirical Bayes approach"""
    
    individual_utils = {}
    
    for respondent in df_long['Respondent_ID'].unique():
        resp_data = df_long[df_long['Respondent_ID'] == respondent]
        
        # Simple empirical Bayes: Start with aggregate, adjust based on individual choices
        resp_utils = {}
        
        for attr, levels in attributes.items():
            resp_utils[attr] = {}
            
            for level in levels:
                # Get aggregate utility as prior
                agg_util = aggregate_utilities[attr][level]
                
                # Count choices for this level
                level_data = resp_data[resp_data[attr] == level]
                n_shown = len(level_data)
                n_chosen = level_data['Choice'].sum()
                
                if n_shown > 0:
                    # Simple adjustment: weight aggregate with individual choice rate
                    choice_rate = n_chosen / n_shown
                    # Bayesian update (simplified)
                    weight = 0.7  # Weight toward aggregate
                    adjusted_util = weight * agg_util + (1 - weight) * (choice_rate - 0.33) * 2
                    resp_utils[attr][level] = adjusted_util
                else:
                    resp_utils[attr][level] = agg_util
        
        individual_utils[respondent] = resp_utils
    
    return individual_utils

def predict_shares(product_configs, utilities):
    """Predict preference shares for configured products"""
    
    # Calculate total utility for each product
    product_utilities = []
    
    for config in product_configs:
        total_utility = sum(
            utilities[attr][level] 
            for attr, level in config.items()
            if attr in utilities and level in utilities[attr]
        )
        product_utilities.append(total_utility)
    
    # Calculate shares using logit choice rule
    shares = softmax(product_utilities) * 100
    
    return shares

def create_attribute_importance_chart(importance):
    """Create bar chart for attribute importance"""
    
    df = pd.DataFrame({
        'Attribute': list(importance.keys()),
        'Importance (%)': list(importance.values())
    }).sort_values('Importance (%)', ascending=True)
    
    fig = go.Figure(go.Bar(
        y=df['Attribute'],
        x=df['Importance (%)'],
        orientation='h',
        marker=dict(
            color=df['Importance (%)'],
            colorscale='Blues',
            showscale=False
        ),
        text=df['Importance (%)'].round(1),
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Relative Importance of Attributes',
        xaxis_title='Importance (%)',
        yaxis_title='',
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
    )
    
    return fig

def create_utilities_chart(utilities, attribute):
    """Create bar chart for part-worth utilities within an attribute"""
    
    levels = list(utilities[attribute].keys())
    values = list(utilities[attribute].values())
    
    df = pd.DataFrame({
        'Level': levels,
        'Utility': values
    }).sort_values('Utility', ascending=True)
    
    # Color positive values differently from negative
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in df['Utility']]
    
    fig = go.Figure(go.Bar(
        y=df['Level'],
        x=df['Utility'],
        orientation='h',
        marker=dict(color=colors),
        text=df['Utility'].round(3),
        texttemplate='%{text}',
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Part-Worth Utilities: {attribute}',
        xaxis_title='Utility',
        yaxis_title='',
        height=300,
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray', zeroline=True, zerolinewidth=2, zerolinecolor='black'),
    )
    
    return fig

# ==================== MAIN APP ====================

st.markdown('<p class="main-header">📊 CBC Conjoint Analysis Platform</p>', unsafe_allow_html=True)
st.markdown("**Comprehensive tool for Choice-Based Conjoint data processing, analysis, and simulation**")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📁 Data Input", 
    "🔧 Model Estimation", 
    "📊 Results & Insights",
    "🎯 Preference Share Simulator",
    "💾 Export Results"
])

# Sidebar for attribute configuration
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Attribute Configuration")

# Allow customization of attributes
use_default = st.sidebar.checkbox("Use default lip balm attributes", value=True)

if use_default:
    attributes = DEFAULT_ATTRIBUTES
else:
    st.sidebar.info("Custom attribute configuration coming soon!")
    attributes = DEFAULT_ATTRIBUTES

# ==================== PAGE 1: DATA INPUT ====================
if page == "📁 Data Input":
    st.markdown('<p class="sub-header">Data Input & Validation</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Upload Data")
        data_source = st.radio("Select data source:", ["Upload Excel File", "Generate Simulated Data"])
        
        if data_source == "Upload Excel File":
            uploaded_file = st.file_uploader("Upload your CBC data (Excel)", type=['xlsx', 'xls'])
            
            if uploaded_file:
                df = pd.read_excel(uploaded_file)
                is_valid, message = validate_data(df)
                
                if is_valid:
                    st.success(message)
                    st.session_state.raw_data = df
                else:
                    st.error(message)
                    df = None
        else:
            st.info("Generating simulated data for testing...")
            n_respondents = st.slider("Number of respondents", 50, 500, 100)
            n_questions = st.slider("Number of questions", 8, 20, 12)
            
            if st.button("Generate Data"):
                df = generate_simulated_data(n_respondents, n_questions, attributes)
                st.session_state.raw_data = df
                st.success(f"Generated data for {n_respondents} respondents!")
    
    with col2:
        st.subheader("📋 Required Data Format")
        st.markdown("""
        Your Excel file should have the following columns:
        - `Respondent_ID`: Unique identifier for each participant
        - `Question`: Question number (1 to N)
        - `Selected_Profile`: Which profile was chosen (1, 2, or 3)
        - `Profile_1_[Attribute]`: Attribute level for profile 1
        - `Profile_2_[Attribute]`: Attribute level for profile 2
        - `Profile_3_[Attribute]`: Attribute level for profile 3
        
        **Example attributes**: Brand, Price, Hero_Claim, Ingredients, SPF
        """)
        
        if st.button("Download Example Template"):
            example_df = create_example_template()
            buffer = BytesIO()
            example_df.to_excel(buffer, index=False)
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Template",
                data=buffer,
                file_name="cbc_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Preview data
    if 'raw_data' in st.session_state:
        st.markdown("---")
        st.subheader("👀 Data Preview")
        
        df = st.session_state.raw_data
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Respondents", df['Respondent_ID'].nunique())
        col2.metric("Total Questions", df['Question'].nunique())
        col3.metric("Total Responses", len(df))
        col4.metric("Attributes", len(attributes))
        
        st.dataframe(df.head(20), use_container_width=True)
        
        # Show attribute levels detected
        st.subheader("🏷️ Detected Attribute Levels")
        for attr in attributes.keys():
            profile_cols = [f'Profile_1_{attr}', f'Profile_2_{attr}', f'Profile_3_{attr}']
            if all(col in df.columns for col in profile_cols):
                levels = set()
                for col in profile_cols:
                    levels.update(df[col].unique())
                st.write(f"**{attr}**: {', '.join(sorted(levels))}")

# ==================== PAGE 2: MODEL ESTIMATION ====================
elif page == "🔧 Model Estimation":
    st.markdown('<p class="sub-header">Model Estimation & Processing</p>', unsafe_allow_html=True)
    
    if 'raw_data' not in st.session_state:
        st.warning("⚠️ Please upload or generate data first in the 'Data Input' page.")
    else:
        df = st.session_state.raw_data
        
        st.info("""
        **Model**: Multinomial Logit (Conditional Logit)  
        **Coding**: Effects coding (sum-to-zero constraint)  
        **Estimation**: Maximum Likelihood
        """)
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Processing data and estimating model..."):
                
                # Step 1: Reshape data
                progress_bar = st.progress(0)
                st.write("**Step 1/4**: Reshaping data to long format...")
                df_long = reshape_data_to_long(df, attributes)
                st.session_state.processed_data = df_long
                progress_bar.progress(25)
                
                # Step 2: Fit model
                st.write("**Step 2/4**: Estimating aggregate utilities...")
                results = fit_conditional_logit(df_long, attributes)
                st.session_state.model_results = results
                progress_bar.progress(50)
                
                # Step 3: Calculate attribute importance
                st.write("**Step 3/4**: Calculating attribute importance...")
                importance, ranges = calculate_attribute_importance(results['utilities'])
                st.session_state.importance = importance
                st.session_state.ranges = ranges
                progress_bar.progress(75)
                
                # Step 4: Calculate individual utilities
                st.write("**Step 4/4**: Calculating individual-level utilities...")
                individual_utils = calculate_individual_utilities(
                    df_long, attributes, results['utilities']
                )
                st.session_state.individual_utilities = individual_utils
                progress_bar.progress(100)
                
                st.success("✅ Analysis complete!")
        
        # Show results if available
        if st.session_state.model_results:
            st.markdown("---")
            st.subheader("📈 Model Summary")
            
            results = st.session_state.model_results
            
            col1, col2 = st.columns(2)
            col1.metric("Log-Likelihood", f"{results['log_likelihood']:.2f}")
            col2.metric("Number of Parameters", len(results['coefficients']))
            
            # Show coefficients
            with st.expander("View Model Coefficients"):
                coef_df = pd.DataFrame({
                    'Parameter': list(results['coefficients'].keys()),
                    'Coefficient': list(results['coefficients'].values())
                }).sort_values('Coefficient', ascending=False)
                st.dataframe(coef_df, use_container_width=True)

# ==================== PAGE 3: RESULTS & INSIGHTS ====================
elif page == "📊 Results & Insights":
    st.markdown('<p class="sub-header">Analysis Results & Insights</p>', unsafe_allow_html=True)
    
    if st.session_state.model_results is None:
        st.warning("⚠️ Please run the analysis first in the 'Model Estimation' page.")
    else:
        utilities = st.session_state.model_results['utilities']
        importance = st.session_state.importance
        
        # Attribute Importance
        st.subheader("1️⃣ Attribute Importance Analysis")
        st.markdown("*Shows which attributes drive purchase decisions*")
        
        fig_importance = create_attribute_importance_chart(importance)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Show importance table
        importance_df = pd.DataFrame({
            'Attribute': list(importance.keys()),
            'Importance (%)': [f"{v:.1f}%" for v in importance.values()],
            'Range': [f"{st.session_state.ranges[k]:.3f}" for k in importance.keys()]
        }).sort_values('Importance (%)', ascending=False)
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Part-Worth Utilities by Attribute
        st.subheader("2️⃣ Part-Worth Utilities by Attribute")
        st.markdown("*Shows preference for each level within attributes*")
        
        # Create tabs for each attribute
        tabs = st.tabs(list(attributes.keys()))
        
        for i, (attr, tab) in enumerate(zip(attributes.keys(), tabs)):
            with tab:
                fig_util = create_utilities_chart(utilities, attr)
                st.plotly_chart(fig_util, use_container_width=True)
                
                # Show utility table
                util_df = pd.DataFrame({
                    'Level': list(utilities[attr].keys()),
                    'Utility': list(utilities[attr].values())
                }).sort_values('Utility', ascending=False)
                util_df['Utility'] = util_df['Utility'].round(4)
                st.dataframe(util_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Individual-Level Summary
        st.subheader("3️⃣ Individual-Level Utilities Summary")
        
        if st.session_state.individual_utilities:
            individual_utils = st.session_state.individual_utilities
            
            # Let user select an attribute to view distribution
            selected_attr = st.selectbox("Select attribute to view distribution:", list(attributes.keys()))
            
            # Collect utilities across all respondents for this attribute
            level_distributions = {level: [] for level in attributes[selected_attr]}
            
            for resp_utils in individual_utils.values():
                for level, util in resp_utils[selected_attr].items():
                    level_distributions[level].append(util)
            
            # Create box plot
            fig = go.Figure()
            for level, utils in level_distributions.items():
                fig.add_trace(go.Box(
                    y=utils,
                    name=level,
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title=f'Distribution of Individual Utilities: {selected_attr}',
                yaxis_title='Utility',
                xaxis_title='Level',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 4: PREFERENCE SHARE SIMULATOR ====================
elif page == "🎯 Preference Share Simulator":
    st.markdown('<p class="sub-header">Preference Share Simulator</p>', unsafe_allow_html=True)
    
    if st.session_state.model_results is None:
        st.warning("⚠️ Please run the analysis first in the 'Model Estimation' page.")
    else:
        utilities = st.session_state.model_results['utilities']
        
        st.markdown("""
        Configure up to 5 products and see their predicted preference shares based on the conjoint model.
        """)
        
        # Number of products to simulate
        n_products = st.slider("Number of products to simulate", 2, 5, 3)
        
        # Create configuration interface
        st.subheader("Configure Products")
        
        product_configs = []
        cols = st.columns(n_products)
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"**Product {i+1}**")
                config = {}
                
                for attr, levels in attributes.items():
                    config[attr] = st.selectbox(
                        attr,
                        options=levels,
                        key=f"product_{i}_{attr}"
                    )
                
                product_configs.append(config)
        
        st.markdown("---")
        
        # Calculate and display shares
        if st.button("Calculate Preference Shares", type="primary"):
            shares = predict_shares(product_configs, utilities)
            
            st.subheader("📊 Predicted Preference Shares")
            
            # Create bar chart
            fig = go.Figure(go.Bar(
                x=[f"Product {i+1}" for i in range(n_products)],
                y=shares,
                marker=dict(
                    color=shares,
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"{s:.1f}%" for s in shares],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Preference Share Simulation',
                yaxis_title='Preference Share (%)',
                xaxis_title='',
                height=400,
                plot_bgcolor='white',
                yaxis=dict(gridcolor='lightgray', range=[0, max(shares) * 1.2]),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed results
            st.subheader("Detailed Results")
            
            results_data = []
            for i, (config, share) in enumerate(zip(product_configs, shares)):
                result_row = {'Product': f'Product {i+1}', 'Preference Share': f'{share:.2f}%'}
                result_row.update(config)
                results_data.append(result_row)
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Calculate total utility for each product
            st.subheader("Total Utility Breakdown")
            for i, config in enumerate(product_configs):
                with st.expander(f"Product {i+1} - Utility Breakdown"):
                    breakdown = []
                    total = 0
                    for attr, level in config.items():
                        util = utilities[attr][level]
                        breakdown.append({'Attribute': attr, 'Level': level, 'Utility': util})
                        total += util
                    
                    breakdown_df = pd.DataFrame(breakdown)
                    breakdown_df['Utility'] = breakdown_df['Utility'].round(4)
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                    st.metric("Total Utility", f"{total:.4f}")

# ==================== PAGE 5: EXPORT RESULTS ====================
elif page == "💾 Export Results":
    st.markdown('<p class="sub-header">Export Results</p>', unsafe_allow_html=True)
    
    if st.session_state.model_results is None:
        st.warning("⚠️ Please run the analysis first in the 'Model Estimation' page.")
    else:
        st.markdown("Export your analysis results in various formats.")
        
        # Export individual utilities
        st.subheader("1. Individual-Level Part-Worth Utilities")
        
        if st.session_state.individual_utilities:
            # Convert to dataframe
            individual_utils = st.session_state.individual_utilities
            
            export_data = []
            for resp_id, resp_utils in individual_utils.items():
                row = {'Respondent_ID': resp_id}
                for attr, levels in resp_utils.items():
                    for level, util in levels.items():
                        row[f'{attr}_{level}'] = util
                export_data.append(row)
            
            export_df = pd.DataFrame(export_data)
            
            # Create Excel file
            buffer = BytesIO()
            export_df.to_excel(buffer, index=False, sheet_name='Individual_Utilities')
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Individual Utilities (Excel)",
                data=buffer,
                file_name="individual_utilities.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.markdown("---")
        
        # Export aggregate results
        st.subheader("2. Aggregate Results Summary")
        
        utilities = st.session_state.model_results['utilities']
        importance = st.session_state.importance
        
        # Create summary workbook
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Sheet 1: Attribute Importance
            importance_df = pd.DataFrame({
                'Attribute': list(importance.keys()),
                'Importance_Percent': list(importance.values()),
                'Range': list(st.session_state.ranges.values())
            })
            importance_df.to_excel(writer, sheet_name='Attribute_Importance', index=False)
            
            # Sheet 2: Part-Worth Utilities
            utilities_data = []
            for attr, levels in utilities.items():
                for level, util in levels.items():
                    utilities_data.append({
                        'Attribute': attr,
                        'Level': level,
                        'Utility': util
                    })
            utilities_df = pd.DataFrame(utilities_data)
            utilities_df.to_excel(writer, sheet_name='Part_Worth_Utilities', index=False)
            
            # Sheet 3: Model Coefficients
            coef_df = pd.DataFrame({
                'Parameter': list(st.session_state.model_results['coefficients'].keys()),
                'Coefficient': list(st.session_state.model_results['coefficients'].values())
            })
            coef_df.to_excel(writer, sheet_name='Model_Coefficients', index=False)
        
        buffer.seek(0)
        
        st.download_button(
            label="📥 Download Aggregate Results (Excel)",
            data=buffer,
            file_name="cbc_aggregate_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.markdown("---")
        st.success("✅ All deliverables ready for export!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**CBC Conjoint Analysis v1.0**")
st.sidebar.markdown("Built with Streamlit & Python")
