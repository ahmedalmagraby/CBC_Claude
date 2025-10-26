import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import io
import re
from typing import List, Dict, Any

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="CBC Conjoint Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem; font-weight: 600; color: #2c3e50; margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'data_processed' not in st.session_state:
    st.session_state.update({
        'data_processed': False,
        'utilities': None,
        'raw_data': None,
        'processed_data': None,
        'attributes': {},,
        'attribute_names': [],
        'price_attribute': None
    })

# ============================================================================
# HELPER FUNCTIONS (WITH IMPROVEMENTS & CACHING)
# ============================================================================

def parse_profile(profile_str: str) -> List[str]:
    """Parse profile string like '[Brand, "1,000 GB", Price]' into a list."""
    if pd.isna(profile_str) or not isinstance(profile_str, str) or not profile_str.strip():
        return None
    
    # CRITICAL FIX: Normalize non-standard delimiters to standard commas
    profile_str = profile_str.replace('’', ',').replace('‘', ',')

    # Remove outer brackets
    profile_str = profile_str.strip()
    if profile_str.startswith('[') and profile_str.endswith(']'):
        profile_str = profile_str[1:-1]
    
    # Regex to split by comma, but ignore commas inside quotes
    values = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', profile_str)
    cleaned_values = [v.strip().strip('"') for v in values if v.strip()]
    
    return cleaned_values if cleaned_values else None

@st.cache_data
def process_cbc_data(df: pd.DataFrame, num_attributes: int) -> pd.DataFrame:
    """Process raw CBC data into a format suitable for modeling."""
    processed_rows = []
    parsing_errors = []
    
    for idx, row in df.iterrows():
        try:
            respondent_id = row['Respondent ID']
            choice_set = row['Set']
            
            selected = parse_profile(row['Selected Profiles'])
            if selected is None or len(selected) != num_attributes:
                parsing_errors.append(f"Row {idx+2}: Invalid 'Selected' profile. Expected {num_attributes} attributes, got {len(selected) if selected else 0}.")
                continue
            
            not_selected_str = str(row['Not Selected Profiles']) if pd.notna(row['Not Selected Profiles']) else ''
            bracket_pattern = r'\[([^\]]+)\]'
            not_selected_matches = re.findall(bracket_pattern, not_selected_str)
            
            not_selected_list = []
            for match in not_selected_matches:
                values = parse_profile(f"[{match}]")
                if values and len(values) == num_attributes:
                    not_selected_list.append(values)
            
            row_data = {'Respondent_ID': respondent_id, 'Choice_Set': choice_set, 'Choice': 1}
            for attr_idx, level in enumerate(selected):
                row_data[f'Attribute_{attr_idx+1}'] = level
            processed_rows.append(row_data)
            
            for not_selected in not_selected_list:
                row_data = {'Respondent_ID': respondent_id, 'Choice_Set': choice_set, 'Choice': 0}
                for attr_idx, level in enumerate(not_selected):
                    row_data[f'Attribute_{attr_idx+1}'] = level
                processed_rows.append(row_data)
        except Exception as e:
            parsing_errors.append(f"Row {idx+2}: Unexpected error - {e}")

    if parsing_errors:
        st.warning(f"Found {len(parsing_errors)} parsing errors. Some rows may have been skipped. Example: {parsing_errors[0]}")
        
    return pd.DataFrame(processed_rows)

def create_design_matrix(df: pd.DataFrame, attributes_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """Create a dummy-coded design matrix for conjoint analysis."""
    design_df = df.copy()
    for attr_name, levels in attributes_dict.items():
        for level in levels[1:]:  # First level is the reference
            col_name = f"{attr_name}_{level}"
            design_df[col_name] = (df[attr_name] == level).astype(int)
    return design_df

@st.cache_data
def estimate_utilities(_design_df: pd.DataFrame, _attributes_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """Estimate part-worth utilities using logistic regression at the individual level."""
    respondents = _design_df['Respondent_ID'].unique()
    utilities_list = []
    failed_respondents = []

    feature_cols = [f"{attr}_{level}" for attr, levels in _attributes_dict.items() for level in levels[1:]]
    
    for resp_id in respondents:
        resp_data = _design_df[_design_df['Respondent_ID'] == resp_id]
        X = resp_data[feature_cols].values
        y = resp_data['Choice'].values
        
        if len(np.unique(y)) < 2:
            failed_respondents.append(resp_id)
            continue

        try:
            model = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs', C=1e6)
            model.fit(X, y)
            coeffs = model.coef_[0]
            
            utility_dict = {'Respondent_ID': resp_id}
            idx = 0
            for attr_name, levels in _attributes_dict.items():
                utility_dict[f"{attr_name}_{levels[0]}"] = 0.0
                for level in levels[1:]:
                    utility_dict[f"{attr_name}_{level}"] = coeffs[idx]
                    idx += 1
            utilities_list.append(utility_dict)
        except Exception:
            failed_respondents.append(resp_id)

    if failed_respondents:
        st.warning(f"Could not estimate utilities for {len(failed_respondents)} respondents (e.g., no variation in choices). They were excluded.")

    return pd.DataFrame(utilities_list)

@st.cache_data
def calculate_attribute_importance(_utilities_df: pd.DataFrame, _attributes_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """Calculate attribute importance using the range method."""
    importance_data = []
    for attr_name, levels in _attributes_dict.items():
        util_cols = [f"{attr_name}_{level}" for level in levels]
        ranges = _utilities_df[util_cols].max(axis=1) - _utilities_df[util_cols].min(axis=1)
        importance_data.append({'Attribute': attr_name, 'Average_Range': ranges.mean()})
    
    importance_df = pd.DataFrame(importance_data)
    total_range = importance_df['Average_Range'].sum()
    if total_range > 0:
        importance_df['Importance_%'] = (importance_df['Average_Range'] / total_range) * 100
    else:
        importance_df['Importance_%'] = 0
    return importance_df.sort_values('Importance_%', ascending=False)

@st.cache_data
def calculate_level_performance(_utilities_df: pd.DataFrame, _attributes_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """Calculate average utility for each level within each attribute."""
    performance_data = []
    for attr_name, levels in _attributes_dict.items():
        for level in levels:
            col_name = f"{attr_name}_{level}"
            performance_data.append({
                'Attribute': attr_name, 'Level': level,
                'Mean_Utility': _utilities_df[col_name].mean(),
                'Std_Utility': _utilities_df[col_name].std()
            })
    return pd.DataFrame(performance_data)

def predict_share(product_configs: List[Dict[str, Any]], utilities_df: pd.DataFrame, attributes_dict: Dict[str, List[str]]) -> np.ndarray:
    """Predict market share for given product configurations using the logit choice rule."""
    n_respondents = len(utilities_df)
    n_products = len(product_configs)
    
    product_utilities = np.zeros((n_respondents, n_products))
    
    for prod_idx, config in enumerate(product_configs):
        total_util_for_product = pd.Series(0.0, index=utilities_df.index)
        for attr_name, level in config.items():
            col_name = f"{attr_name}_{level}"
            if col_name in utilities_df.columns:
                total_util_for_product += utilities_df[col_name]
        product_utilities[:, prod_idx] = total_util_for_product.values

    exp_utils = np.exp(product_utilities)
    sum_exp_utils_per_respondent = exp_utils.sum(axis=1, keepdims=True)
    sum_exp_utils_per_respondent[sum_exp_utils_per_respondent == 0] = 1 
    probabilities = exp_utils / sum_exp_utils_per_respondent
    shares = probabilities.mean(axis=0)
    
    return shares

# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

st.sidebar.markdown("## 📊 CBC Analysis Platform")
page = st.sidebar.radio(
    "Navigate",
    ["📤 Data Upload", "🧮 Model Estimation", "📈 Results & Insights", 
     "💰 Price Optimization", "🎮 Market Simulator"],
    label_visibility="collapsed"
)

if page == "📤 Data Upload":
    st.markdown("<div class='main-header'>📤 Data Upload & Processing</div>", unsafe_allow_html=True)
    st.markdown("""
    Upload your CBC survey data (Excel/CSV) with columns: `Respondent ID`, `Set`, `Selected Profiles`, and `Not Selected Profiles`. 
    Profile format should be `[Level1, Level2, ...]`.
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Data File", type=['xlsx', 'xls', 'csv'])
    with col2:
        num_attributes = st.number_input("Number of Attributes", min_value=2, max_value=10, value=3, key="num_attr")
        attribute_names_input = st.text_area("Attribute Names (one per line)", value="Brand\nStorage\nPrice", height=150, key="attr_names")
    
    attribute_names = [name.strip() for name in attribute_names_input.strip().split('\n') if name.strip()]
    price_attribute_name = None
    if attribute_names and len(attribute_names) == num_attributes:
        price_attribute_name = st.selectbox(
            "Identify the 'Price' Attribute for Cleaning",
            options=attribute_names,
            index=len(attribute_names) - 1,
            help="This will be used to clean currency symbols and formatting from the price levels."
        )

    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.raw_data = raw_df
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully! Found {len(raw_df)} rows.")
            
            with st.expander("📋 View Raw Data"):
                st.dataframe(raw_df.head(10))
            
            if len(attribute_names) != num_attributes:
                st.error(f"❌ Please provide exactly {num_attributes} attribute names!")
            else:
                st.session_state.attribute_names = attribute_names
                st.session_state.price_attribute = price_attribute_name
                
                if st.button("🚀 Process Data", type="primary"):
                    with st.spinner("Processing data..."):
                        processed_df = process_cbc_data(raw_df, num_attributes)
                        
                        if processed_df.empty:
                            st.error("❌ No data was processed! Check file format, attribute count, and profile formatting.")
                        else:
                            rename_map = {f'Attribute_{i+1}': name for i, name in enumerate(attribute_names)}
                            processed_df.rename(columns=rename_map, inplace=True)
                            
                            # IMPROVEMENT: Clean the designated price column before detecting levels
                            if price_attribute_name and price_attribute_name in processed_df.columns:
                                processed_df[price_attribute_name] = (
                                    processed_df[price_attribute_name].astype(str)
                                    .str.replace(r'[$,]', '', regex=True) # Remove $ and commas
                                    .str.strip()
                                )
                                st.info(f"Cleaned the '{price_attribute_name}' column by removing currency symbols and commas.")

                            # Detect unique levels from the (potentially cleaned) data
                            attributes_dict = {}
                            for name in attribute_names:
                                levels = processed_df[name].dropna().astype(str).unique().tolist()
                                # Try to sort numerically if possible, otherwise sort alphabetically
                                try:
                                    attributes_dict[name] = sorted(levels, key=float)
                                except ValueError:
                                    attributes_dict[name] = sorted(levels)

                            st.session_state.update({
                                'attributes': attributes_dict,
                                'processed_data': processed_df,
                                'data_processed': True
                            })
                            st.success("✅ Data processed successfully!")
                            st.dataframe(processed_df.head(20))
                            
                            st.markdown("### ✅ Detected Attribute Levels")
                            for attr_name, levels in attributes_dict.items():
                                with st.expander(f"**{attr_name}** ({len(levels)} levels)", expanded=True):
                                    st.write(", ".join(map(str, levels)) if levels else "⚠️ No levels detected!")
                            st.info("👉 Proceed to 'Model Estimation' to calculate utilities!")
        except Exception as e:
            st.error(f"An error occurred while reading or processing the file: {e}")

elif page == "🧮 Model Estimation":
    st.markdown("<div class='main-header'>🧮 Model Estimation</div>", unsafe_allow_html=True)
    if not st.session_state.data_processed:
        st.warning("⚠️ Please upload and process data on the 'Data Upload' page first.")
    else:
        st.markdown("""
        ### Individual-Level Multinomial Logit Model
        This analysis uses logistic regression for each respondent to estimate part-worth utilities, capturing individual preferences.
        """)
        processed_df = st.session_state.processed_data
        attributes_dict = st.session_state.attributes
        
        design_df = create_design_matrix(processed_df, attributes_dict)
        
        if st.button("⚡ Run Model Estimation", type="primary"):
            with st.spinner("Estimating utilities... This may take a moment."):
                utilities_df = estimate_utilities(design_df, attributes_dict)
                st.session_state.utilities = utilities_df
                
                st.success("✅ Model estimation completed!")
                st.dataframe(utilities_df.head(10))
                st.info("👉 Proceed to 'Results & Insights' to view analysis!")

elif page == "📈 Results & Insights":
    st.markdown("<div class='main-header'>📈 Results & Insights</div>", unsafe_allow_html=True)
    if st.session_state.utilities is None:
        st.warning("⚠️ Please run model estimation on the 'Model Estimation' page first.")
    else:
        utilities_df = st.session_state.utilities
        attributes_dict = st.session_state.attributes
        
        importance_df = calculate_attribute_importance(utilities_df, attributes_dict)
        performance_df = calculate_level_performance(utilities_df, attributes_dict)
        
        st.markdown("## 🎯 Attribute Importance")
        fig_importance = px.bar(importance_df, y='Attribute', x='Importance_%', orientation='h',
                                text='Importance_%', title="Attribute Importance (% of Total)")
        fig_importance.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("## 📊 Level Performance")
        for attr_name in attributes_dict.keys():
            attr_data = performance_df[performance_df['Attribute'] == attr_name].sort_values('Mean_Utility')
            fig_level = px.bar(attr_data, y='Level', x='Mean_Utility', orientation='h',
                               error_x='Std_Utility', title=f"{attr_name} - Level Utilities",
                               color='Mean_Utility', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_level, use_container_width=True)

elif page == "💰 Price Optimization":
    st.markdown("<div class='main-header'>💰 Price Optimization</div>", unsafe_allow_html=True)
    if st.session_state.utilities is None:
        st.warning("⚠️ Please run model estimation first.")
    elif st.session_state.price_attribute is None:
        st.error("A price attribute was not identified. Please re-process your data on the 'Data Upload' page.")
    else:
        utilities_df = st.session_state.utilities
        attributes_dict = st.session_state.attributes
        price_attr = st.session_state.price_attribute
        
        st.info(f"Using **'{price_attr}'** as the price attribute.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Your Product Configuration")
            product_config = {
                attr: st.selectbox(f"{attr}", levels, key=f"opt_{attr}")
                for attr, levels in attributes_dict.items() if attr != price_attr
            }
        with col2:
            st.markdown("#### Competitor Product")
            competitor_config = {
                attr: st.selectbox(f"{attr}", levels, key=f"comp_{attr}")
                for attr, levels in attributes_dict.items()
            }

        if st.button("🔍 Analyze Price Points", type="primary"):
            with st.spinner("Analyzing prices..."):
                price_levels = attributes_dict[price_attr]
                results = []
                for price in price_levels:
                    test_config = {**product_config, price_attr: price}
                    configs = [test_config, competitor_config]
                    shares = predict_share(configs, utilities_df, attributes_dict)
                    
                    try:
                        price_numeric = float(re.sub(r'[^\d.]', '', str(price)))
                    except (ValueError, TypeError):
                        price_numeric = 0

                    results.append({
                        'Price': price, 'Price_Numeric': price_numeric,
                        'Preference_Share': shares[0] * 100, 'Revenue_Index': shares[0] * price_numeric
                    })
                
                results_df = pd.DataFrame(results).sort_values('Price_Numeric')
                
                if not results_df.empty:
                    st.subheader("Optimal Price Points")
                    c1, c2 = st.columns(2)
                    with c1: st.metric("📈 Max Share Price", results_df.loc[results_df['Preference_Share'].idxmax(), 'Price'])
                    with c2: st.metric("💵 Max Revenue Price", results_df.loc[results_df['Revenue_Index'].idxmax(), 'Price'])
                    
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(x=results_df['Price_Numeric'], y=results_df['Preference_Share'], name='Preference Share (%)'))
                    fig_price.add_trace(go.Scatter(x=results_df['Price_Numeric'], y=results_df['Revenue_Index'], name='Revenue Index', yaxis='y2'))
                    fig_price.update_layout(title="Price Elasticity Analysis", xaxis_title="Price", yaxis_title="Preference Share (%)",
                                            yaxis2=dict(title="Revenue Index", overlaying='y', side='right'), hovermode='x unified')
                    st.plotly_chart(fig_price, use_container_width=True)
                    st.dataframe(results_df)

elif page == "🎮 Market Simulator":
    st.markdown("<div class='main-header'>🎮 Market Simulator</div>", unsafe_allow_html=True)
    if st.session_state.utilities is None:
        st.warning("⚠️ Please run model estimation first.")
    else:
        utilities_df = st.session_state.utilities
        attributes_dict = st.session_state.attributes
        
        num_products = st.slider("Number of Products", 2, 5, 3)
        product_configs = []
        cols = st.columns(num_products)
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"#### Product {i+1}")
                config = {attr: st.selectbox(f"{attr}", levels, key=f"sim_p{i}_{attr}")
                          for attr, levels in attributes_dict.items()}
                product_configs.append(config)
        
        if st.button("🎯 Calculate Preference Shares", type="primary"):
            shares = predict_share(product_configs, utilities_df, attributes_dict) * 100
            
            st.markdown("### 📊 Preference Shares")
            metric_cols = st.columns(num_products)
            for i, (col, share) in enumerate(zip(metric_cols, shares)):
                with col: st.metric(f"Product {i+1}", f"{share:.1f}%")

            results_df = pd.DataFrame([{ 'Product': f'Product {i+1}', 'Share_%': share, **config }
                                       for i, (config, share) in enumerate(zip(product_configs, shares))])
            
            fig_pie = px.pie(results_df, values='Share_%', names='Product', title='Market Share Distribution', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.dataframe(results_df.style.format({'Share_%': '{:.2f}%'}))

# Footer
st.sidebar.markdown("---")
st.sidebar.info("CBC Analysis Platform | Built with Streamlit & Python")
