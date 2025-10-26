import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from itertools import product
import io
import re

# Page configuration
st.set_page_config(
    page_title="CBC Conjoint Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'utilities' not in st.session_state:
    st.session_state.utilities = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'attributes' not in st.session_state:
    st.session_state.attributes = {}
if 'attribute_names' not in st.session_state:
    st.session_state.attribute_names = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_profile(profile_str):
    """Parse profile string like '[Brand, Storage, Price]' into list"""
    if pd.isna(profile_str) or profile_str == '' or profile_str is None:
        return None
    
    # Convert to string and clean
    profile_str = str(profile_str).strip()
    
    # Remove outer brackets
    if profile_str.startswith('[') and profile_str.endswith(']'):
        profile_str = profile_str[1:-1]
    
    # Split by comma and clean each value
    values = [item.strip() for item in profile_str.split(',') if item.strip()]
    
    return values if values else None

def process_cbc_data(df, num_attributes):
    """Process raw CBC data into format suitable for modeling"""
    processed_rows = []
    
    for idx, row in df.iterrows():
        respondent_id = row['Respondent ID']
        choice_set = row['Set']
        
        # Parse selected profile
        selected = parse_profile(row['Selected Profiles'])
        
        # Skip if selected profile is invalid
        if selected is None or len(selected) != num_attributes:
            continue
        
        # Parse not selected profiles
        not_selected_str = str(row['Not Selected Profiles']) if pd.notna(row['Not Selected Profiles']) else ''
        
        # Split by semicolon for multiple alternatives
        not_selected_raw = [p.strip() for p in not_selected_str.split(';') if p.strip()]
        
        # Parse each not selected profile
        not_selected_list = []
        for ns_str in not_selected_raw:
            ns_parsed = parse_profile(ns_str)
            if ns_parsed is not None and len(ns_parsed) == num_attributes:
                not_selected_list.append(ns_parsed)
        
        # Add selected profile (Choice = 1)
        row_data = {
            'Respondent_ID': respondent_id,
            'Choice_Set': choice_set,
            'Choice': 1
        }
        for attr_idx in range(num_attributes):
            row_data[f'Attribute_{attr_idx+1}'] = selected[attr_idx]
        processed_rows.append(row_data)
        
        # Add not selected profiles (Choice = 0)
        for not_selected in not_selected_list:
            row_data = {
                'Respondent_ID': respondent_id,
                'Choice_Set': choice_set,
                'Choice': 0
            }
            for attr_idx in range(num_attributes):
                row_data[f'Attribute_{attr_idx+1}'] = not_selected[attr_idx]
            processed_rows.append(row_data)
    
    result_df = pd.DataFrame(processed_rows)
    
    # Debug: Print summary
    if len(result_df) > 0:
        print(f"Processed {len(result_df)} choice observations")
        print(f"Unique respondents: {result_df['Respondent_ID'].nunique()}")
        print(f"Choices per set: {result_df.groupby(['Respondent_ID', 'Choice_Set']).size().mean():.1f}")
    
    return result_df

def create_design_matrix(df, attributes_dict):
    """Create dummy-coded design matrix for conjoint analysis"""
    design_df = df.copy()
    
    # Create dummy variables for each attribute
    for attr_name, levels in attributes_dict.items():
        # Use the first level as reference (dropped)
        for level in levels[1:]:  # Skip first level (reference)
            col_name = f"{attr_name}_{level}"
            design_df[col_name] = (df[attr_name] == level).astype(int)
    
    return design_df

def estimate_utilities(design_df, attributes_dict):
    """Estimate part-worth utilities using multinomial logit at individual level"""
    respondents = design_df['Respondent_ID'].unique()
    utilities_list = []
    
    # Get feature columns (all dummy variables)
    feature_cols = []
    for attr_name, levels in attributes_dict.items():
        for level in levels[1:]:
            feature_cols.append(f"{attr_name}_{level}")
    
    for resp_id in respondents:
        resp_data = design_df[design_df['Respondent_ID'] == resp_id]
        
        X = resp_data[feature_cols].values
        y = resp_data['Choice'].values
        
        # Fit logistic regression
        try:
            model = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
            model.fit(X, y)
            
            # Get coefficients
            coeffs = model.coef_[0]
            
            # Build utility dictionary
            utility_dict = {'Respondent_ID': resp_id}
            
            # Add utilities for each attribute level
            idx = 0
            for attr_name, levels in attributes_dict.items():
                # Reference level gets utility of 0
                utility_dict[f"{attr_name}_{levels[0]}"] = 0.0
                
                # Other levels get their coefficients
                for level in levels[1:]:
                    utility_dict[f"{attr_name}_{level}"] = coeffs[idx]
                    idx += 1
            
            utilities_list.append(utility_dict)
        except:
            # If model fails, assign zeros
            utility_dict = {'Respondent_ID': resp_id}
            for attr_name, levels in attributes_dict.items():
                for level in levels:
                    utility_dict[f"{attr_name}_{level}"] = 0.0
            utilities_list.append(utility_dict)
    
    return pd.DataFrame(utilities_list)

def calculate_attribute_importance(utilities_df, attributes_dict):
    """Calculate attribute importance using range method"""
    importance_data = []
    
    for attr_name, levels in attributes_dict.items():
        # Get utility columns for this attribute
        util_cols = [f"{attr_name}_{level}" for level in levels]
        
        # Calculate range for each respondent
        ranges = []
        for _, row in utilities_df.iterrows():
            utils = [row[col] for col in util_cols]
            ranges.append(max(utils) - min(utils))
        
        avg_range = np.mean(ranges)
        importance_data.append({
            'Attribute': attr_name,
            'Average_Range': avg_range
        })
    
    importance_df = pd.DataFrame(importance_data)
    total_range = importance_df['Average_Range'].sum()
    importance_df['Importance_%'] = (importance_df['Average_Range'] / total_range) * 100
    
    return importance_df

def calculate_level_performance(utilities_df, attributes_dict):
    """Calculate average utility for each level within each attribute"""
    performance_data = []
    
    for attr_name, levels in attributes_dict.items():
        for level in levels:
            col_name = f"{attr_name}_{level}"
            avg_utility = utilities_df[col_name].mean()
            std_utility = utilities_df[col_name].std()
            
            performance_data.append({
                'Attribute': attr_name,
                'Level': level,
                'Mean_Utility': avg_utility,
                'Std_Utility': std_utility
            })
    
    return pd.DataFrame(performance_data)

def predict_share(product_configs, utilities_df, attributes_dict):
    """Predict market share for given product configurations"""
    n_respondents = len(utilities_df)
    n_products = len(product_configs)
    
    # Calculate total utility for each product for each respondent
    product_utilities = np.zeros((n_respondents, n_products))
    
    for prod_idx, config in enumerate(product_configs):
        for resp_idx, (_, resp_utils) in enumerate(utilities_df.iterrows()):
            total_util = 0
            for attr_name, level in config.items():
                col_name = f"{attr_name}_{level}"
                if col_name in resp_utils:
                    total_util += resp_utils[col_name]
            product_utilities[resp_idx, prod_idx] = total_util
    
    # Apply multinomial logit choice rule
    exp_utils = np.exp(product_utilities)
    shares = exp_utils.sum(axis=0) / exp_utils.sum()
    
    return shares

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("## 📊 CBC Analysis Platform")
page = st.sidebar.radio(
    "Navigate",
    ["📤 Data Upload", "🧮 Model Estimation", "📈 Results & Insights", 
     "💰 Price Optimization", "🎮 Market Simulator"],
    label_visibility="collapsed"
)

# ============================================================================
# PAGE 1: DATA UPLOAD & PROCESSING
# ============================================================================

if page == "📤 Data Upload":
    st.markdown("<div class='main-header'>📤 Data Upload & Processing</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Instructions
    Upload your CBC survey data in Excel format with the following columns:
    - **Respondent ID**: Unique identifier for each participant
    - **Set**: Choice set number (0, 1, 2, ...)
    - **Selected Profiles**: Profile chosen by respondent (format: `[Level1, Level2, Level3, ...]`)
    - **Not Selected Profiles**: Profile(s) not chosen (same format, can be separated by semicolons for multiple)
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    with col2:
        num_attributes = st.number_input("Number of Attributes", min_value=2, max_value=10, value=3)
        attribute_names_input = st.text_area(
            "Attribute Names (one per line)",
            value="Brand\nStorage\nPrice",
            height=150
        )
    
    if uploaded_file is not None:
        # Read the data
        raw_df = pd.read_excel(uploaded_file)
        st.session_state.raw_data = raw_df
        
        st.success(f"✅ File uploaded successfully! Found {len(raw_df)} rows.")
        
        # Show raw data preview
        with st.expander("📋 View Raw Data"):
            st.dataframe(raw_df.head(10))
        
        # Parse attribute names
        attribute_names = [name.strip() for name in attribute_names_input.strip().split('\n') if name.strip()]
        
        if len(attribute_names) != num_attributes:
            st.error(f"❌ Please provide exactly {num_attributes} attribute names!")
        else:
            st.session_state.attribute_names = attribute_names
            
            if st.button("🚀 Process Data", type="primary"):
                with st.spinner("Processing data..."):
                    # Process the data
                    processed_df = process_cbc_data(raw_df, num_attributes)
                    
                    if len(processed_df) == 0:
                        st.error("❌ No data was processed! Please check your data format.")
                        st.stop()
                    
                    # Rename attribute columns with the user-provided names
                    rename_map = {}
                    for i, name in enumerate(attribute_names):
                        rename_map[f'Attribute_{i+1}'] = name
                    processed_df.rename(columns=rename_map, inplace=True)
                    
                    # Detect unique levels for EACH attribute separately
                    attributes_dict = {}
                    
                    st.markdown("### 🔍 Attribute Detection Debug")
                    
                    for i, attr_name in enumerate(attribute_names):
                        if attr_name in processed_df.columns:
                            # Get only the values from THIS column
                            column_values = processed_df[attr_name].dropna().astype(str).str.strip()
                            # Get unique values
                            unique_vals = sorted(column_values.unique().tolist())
                            # Remove empty strings
                            unique_vals = [v for v in unique_vals if v and v != '']
                            
                            attributes_dict[attr_name] = unique_vals
                            
                            # Debug output
                            st.write(f"**Column {i+1}: {attr_name}**")
                            st.write(f"  - Sample values: {column_values.head(5).tolist()}")
                            st.write(f"  - Unique count: {len(unique_vals)}")
                        else:
                            st.error(f"❌ Column '{attr_name}' not found in processed data!")
                            attributes_dict[attr_name] = []
                    
                    st.session_state.attributes = attributes_dict
                    st.session_state.processed_data = processed_df
                    st.session_state.data_processed = True
                    
                    st.success("✅ Data processed successfully!")
                    
                    # Show processed data with better formatting
                    st.markdown("### 📊 Processed Data Preview")
                    display_cols = ['Respondent_ID', 'Choice_Set', 'Choice'] + attribute_names
                    st.dataframe(processed_df[display_cols].head(20), use_container_width=True)
                    
                    # Show attribute levels in expandable sections
                    st.markdown("### ✅ Detected Attribute Levels")
                    
                    for i, (attr_name, levels) in enumerate(attributes_dict.items(), 1):
                        with st.expander(f"**{i}. {attr_name}** ({len(levels)} levels)", expanded=True):
                            if levels:
                                # Display as bullets
                                for level in levels:
                                    st.write(f"  • {level}")
                            else:
                                st.warning("⚠️ No levels detected!")
                    
                    st.info("👉 Proceed to 'Model Estimation' to calculate utilities!")

# ============================================================================
# PAGE 2: MODEL ESTIMATION
# ============================================================================

elif page == "🧮 Model Estimation":
    st.markdown("<div class='main-header'>🧮 Model Estimation</div>", unsafe_allow_html=True)
    
    if not st.session_state.data_processed:
        st.warning("⚠️ Please upload and process data first!")
    else:
        st.markdown("""
        ### Multinomial Logit Model
        This analysis uses individual-level multinomial logit regression to estimate part-worth utilities.
        Each respondent's choices are modeled independently to capture heterogeneity in preferences.
        """)
        
        processed_df = st.session_state.processed_data
        attributes_dict = st.session_state.attributes
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Respondents", processed_df['Respondent_ID'].nunique())
        with col2:
            st.metric("Total Choices", len(processed_df))
        with col3:
            st.metric("Attributes", len(attributes_dict))
        
        if st.button("⚡ Run Model Estimation", type="primary"):
            with st.spinner("Estimating utilities... This may take a moment."):
                # Create design matrix
                design_df = create_design_matrix(processed_df, attributes_dict)
                
                # Estimate utilities
                utilities_df = estimate_utilities(design_df, attributes_dict)
                st.session_state.utilities = utilities_df
                
                st.success("✅ Model estimation completed!")
                
                # Display utilities
                st.markdown("### Part-Worth Utilities (First 10 Respondents)")
                st.dataframe(utilities_df.head(10))
                
                # Summary statistics
                st.markdown("### Utility Statistics")
                util_cols = [col for col in utilities_df.columns if col != 'Respondent_ID']
                summary_stats = utilities_df[util_cols].describe().T
                st.dataframe(summary_stats)
                
                # Download button for utilities
                csv_buffer = io.StringIO()
                utilities_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Utilities (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="part_worth_utilities.csv",
                    mime="text/csv"
                )
                
                st.info("👉 Proceed to 'Results & Insights' to view analysis!")

# ============================================================================
# PAGE 3: RESULTS & INSIGHTS
# ============================================================================

elif page == "📈 Results & Insights":
    st.markdown("<div class='main-header'>📈 Results & Insights</div>", unsafe_allow_html=True)
    
    if st.session_state.utilities is None:
        st.warning("⚠️ Please run model estimation first!")
    else:
        utilities_df = st.session_state.utilities
        attributes_dict = st.session_state.attributes
        
        # Calculate attribute importance
        importance_df = calculate_attribute_importance(utilities_df, attributes_dict)
        
        # Calculate level performance
        performance_df = calculate_level_performance(utilities_df, attributes_dict)
        
        # Attribute Importance Chart
        st.markdown("## 🎯 Attribute Importance")
        st.markdown("Shows the relative importance of each attribute in driving purchase decisions.")
        
        fig_importance = go.Figure(data=[
            go.Bar(
                x=importance_df['Importance_%'],
                y=importance_df['Attribute'],
                orientation='h',
                marker=dict(
                    color=importance_df['Importance_%'],
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Importance %")
                ),
                text=importance_df['Importance_%'].round(1),
                texttemplate='%{text}%',
                textposition='outside'
            )
        ])
        
        fig_importance.update_layout(
            title="Attribute Importance (% of Total)",
            xaxis_title="Importance (%)",
            yaxis_title="",
            height=400,
            showlegend=False,
            yaxis=dict(categoryorder='total ascending')
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Display importance table
        st.dataframe(importance_df.style.format({'Importance_%': '{:.2f}%', 'Average_Range': '{:.3f}'}))
        
        st.markdown("---")
        
        # Level Performance Charts
        st.markdown("## 📊 Level Performance within Attributes")
        st.markdown("Shows the average utility for each level within each attribute.")
        
        # Create separate chart for each attribute
        for attr_name in attributes_dict.keys():
            attr_data = performance_df[performance_df['Attribute'] == attr_name].copy()
            attr_data = attr_data.sort_values('Mean_Utility', ascending=True)
            
            fig_level = go.Figure()
            
            fig_level.add_trace(go.Bar(
                x=attr_data['Mean_Utility'],
                y=attr_data['Level'],
                orientation='h',
                marker=dict(
                    color=attr_data['Mean_Utility'],
                    colorscale='RdYlGn',
                    showscale=False
                ),
                text=attr_data['Mean_Utility'].round(3),
                texttemplate='%{text}',
                textposition='outside',
                error_x=dict(
                    type='data',
                    array=attr_data['Std_Utility'],
                    visible=True
                )
            ))
            
            fig_level.update_layout(
                title=f"{attr_name} - Level Performance",
                xaxis_title="Mean Utility",
                yaxis_title="",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_level, use_container_width=True)
        
        # Download performance data
        csv_buffer = io.StringIO()
        performance_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Level Performance (CSV)",
            data=csv_buffer.getvalue(),
            file_name="level_performance.csv",
            mime="text/csv"
        )

# ============================================================================
# PAGE 4: PRICE OPTIMIZATION
# ============================================================================

elif page == "💰 Price Optimization":
    st.markdown("<div class='main-header'>💰 Price Optimization</div>", unsafe_allow_html=True)
    
    if st.session_state.utilities is None:
        st.warning("⚠️ Please run model estimation first!")
    else:
        utilities_df = st.session_state.utilities
        attributes_dict = st.session_state.attributes
        
        st.markdown("""
        ### Find Optimal Price
        Configure your product and find the price point that maximizes preference share or revenue.
        """)
        
        # Find price attribute
        price_attr = None
        for attr in attributes_dict.keys():
            if 'price' in attr.lower():
                price_attr = attr
                break
        
        if price_attr is None:
            st.error("No price attribute found in the data!")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Product Configuration")
                product_config = {}
                for attr_name, levels in attributes_dict.items():
                    if attr_name != price_attr:
                        product_config[attr_name] = st.selectbox(
                            f"{attr_name}",
                            levels,
                            key=f"opt_{attr_name}"
                        )
            
            with col2:
                st.markdown("#### Competitor Product")
                competitor_config = {}
                for attr_name, levels in attributes_dict.items():
                    competitor_config[attr_name] = st.selectbox(
                        f"{attr_name}",
                        levels,
                        key=f"comp_{attr_name}"
                    )
            
            if st.button("🔍 Analyze Price Points", type="primary"):
                with st.spinner("Analyzing prices..."):
                    price_levels = attributes_dict[price_attr]
                    
                    # Calculate share for each price point
                    results = []
                    for price in price_levels:
                        test_config = product_config.copy()
                        test_config[price_attr] = price
                        
                        configs = [test_config, competitor_config]
                        shares = predict_share(configs, utilities_df, attributes_dict)
                        
                        # Extract numeric price
                        price_numeric = float(re.sub(r'[^\d.]', '', price))
                        
                        results.append({
                            'Price': price,
                            'Price_Numeric': price_numeric,
                            'Preference_Share': shares[0] * 100,
                            'Revenue_Index': shares[0] * price_numeric
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Find optimal price
                    optimal_share_idx = results_df['Preference_Share'].idxmax()
                    optimal_revenue_idx = results_df['Revenue_Index'].idxmax()
                    
                    # Display optimal prices
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Max Share Price",
                            results_df.loc[optimal_share_idx, 'Price'],
                            f"{results_df.loc[optimal_share_idx, 'Preference_Share']:.1f}%"
                        )
                    with col2:
                        st.metric(
                            "Max Revenue Price",
                            results_df.loc[optimal_revenue_idx, 'Price'],
                            f"Index: {results_df.loc[optimal_revenue_idx, 'Revenue_Index']:.1f}"
                        )
                    with col3:
                        avg_share = results_df['Preference_Share'].mean()
                        st.metric("Average Share", f"{avg_share:.1f}%")
                    
                    # Price elasticity curve
                    fig_price = go.Figure()
                    
                    fig_price.add_trace(go.Scatter(
                        x=results_df['Price_Numeric'],
                        y=results_df['Preference_Share'],
                        mode='lines+markers',
                        name='Preference Share',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig_price.add_trace(go.Scatter(
                        x=results_df['Price_Numeric'],
                        y=results_df['Revenue_Index'],
                        mode='lines+markers',
                        name='Revenue Index',
                        line=dict(color='#2ca02c', width=3),
                        marker=dict(size=10),
                        yaxis='y2'
                    ))
                    
                    fig_price.update_layout(
                        title="Price Elasticity Analysis",
                        xaxis_title="Price",
                        yaxis_title="Preference Share (%)",
                        yaxis2=dict(
                            title="Revenue Index",
                            overlaying='y',
                            side='right'
                        ),
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_price, use_container_width=True)
                    
                    # Display table
                    st.dataframe(results_df.style.format({
                        'Preference_Share': '{:.2f}%',
                        'Revenue_Index': '{:.2f}'
                    }))

# ============================================================================
# PAGE 5: MARKET SIMULATOR
# ============================================================================

elif page == "🎮 Market Simulator":
    st.markdown("<div class='main-header'>🎮 Market Simulator</div>", unsafe_allow_html=True)
    
    if st.session_state.utilities is None:
        st.warning("⚠️ Please run model estimation first!")
    else:
        utilities_df = st.session_state.utilities
        attributes_dict = st.session_state.attributes
        
        st.markdown("""
        ### Preference Share Simulator
        Configure up to 5 products and see their predicted market shares based on the conjoint model.
        """)
        
        # Number of products to simulate
        num_products = st.slider("Number of Products", min_value=2, max_value=5, value=3)
        
        # Configure products
        st.markdown("### Configure Products")
        
        product_configs = []
        cols = st.columns(num_products)
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"#### Product {i+1}")
                config = {}
                for attr_name, levels in attributes_dict.items():
                    config[attr_name] = st.selectbox(
                        f"{attr_name}",
                        levels,
                        key=f"sim_p{i}_{attr_name}"
                    )
                product_configs.append(config)
        
        if st.button("🎯 Calculate Preference Shares", type="primary"):
            with st.spinner("Calculating shares..."):
                # Predict shares
                shares = predict_share(product_configs, utilities_df, attributes_dict)
                shares_pct = shares * 100
                
                # Create results dataframe
                results_data = []
                for i, (config, share) in enumerate(zip(product_configs, shares_pct)):
                    result = {'Product': f'Product {i+1}', 'Share_%': share}
                    result.update(config)
                    results_data.append(result)
                
                results_df = pd.DataFrame(results_data)
                
                # Display metrics
                st.markdown("### 📊 Preference Shares")
                metric_cols = st.columns(num_products)
                for i, (col, share) in enumerate(zip(metric_cols, shares_pct)):
                    with col:
                        st.metric(f"Product {i+1}", f"{share:.1f}%")
                
                # Pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[f'Product {i+1}' for i in range(num_products)],
                    values=shares_pct,
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Set3[:num_products])
                )])
                
                fig_pie.update_layout(
                    title="Market Share Distribution",
                    height=500
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Bar chart
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=[f'Product {i+1}' for i in range(num_products)],
                        y=shares_pct,
                        marker=dict(
                            color=shares_pct,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Share %")
                        ),
                        text=shares_pct.round(1),
                        texttemplate='%{text}%',
                        textposition='outside'
                    )
                ])
                
                fig_bar.update_layout(
                    title="Preference Share Comparison",
                    xaxis_title="Product",
                    yaxis_title="Preference Share (%)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Product configuration table
                st.markdown("### Product Configurations")
                st.dataframe(results_df.style.format({'Share_%': '{:.2f}%'}))
                
                # Download results
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Simulation Results (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="market_simulation.csv",
                    mime="text/csv"
                )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p><strong>CBC Analysis Platform</strong></p>
    <p>Built with Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)
