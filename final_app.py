import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="E-commerce Returns Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- HELPER FUNCTIONS ---
@st.cache_data
def get_img_as_base64(file):
    """Encodes an image to Base64 to be used in CSS for the background."""
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_data
def load_data(file_path):
    """Loads and cleans the e-commerce sales data."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.rename(columns={'returned?': 'returned'}, inplace=True)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['returned_numeric'] = df['returned'].astype(str).apply(lambda x: 1 if x.strip().lower() == 'yes' else 0)
    return df

@st.cache_resource
def train_model(df):
    """Trains the RandomForest model and returns the pipeline and feature names."""
    feature_cols = [
        'category', 'payment_method', 'delivery_time_days', 'region',
        'total_amount', 'shipping_cost', 'profit_margin',
        'customer_age', 'customer_gender'
    ]
    X = df[feature_cols]
    y = df['returned']
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=150, max_depth=10)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X, y)

    cat_features_out = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(numerical_features) + list(cat_features_out)

    return pipeline, all_feature_names

def set_bg(image_file):
    """Set background image only for main content (not sidebar)."""
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    page_bg_img = f"""
    <style>
        /* خلفية للجزء الرئيسي فقط (وليس الشريط الجانبي) */
        .stApp {{
            background: none;
        }}

        /* تغليف المحتوى الأساسي */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* نخلي الشريط الجانبي بلون ثابت */
        [data-testid="stSidebar"] {{
            background-color: rgba(30, 30, 30, 0.9) !important; /* تقدر تغيّر اللون حسب رغبتك */
            color: white;
        }}

        /* شفافيات الهيدر إذا حبيت */
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: rgba(0,0,0,0);
        }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

if os.path.exists("background.jpeg"):
    set_bg("background.jpeg")
else:
    st.warning("⚠ الصورة background.jpeg مش موجودة في نفس الفولدر")
# --- DATA LOADING AND MODEL TRAINING ---
DATA_FILE = 'ecommerce_sales_34500.csv'
df = load_data(DATA_FILE)
model_pipeline, feature_names = train_model(df.copy())

# --- Ensure EDA columns exist in both dataframes ---
if 'order_month' not in df.columns:
    df['order_month'] = df['order_date'].dt.month
if 'order_day_of_week' not in df.columns:
    df['order_day_of_week'] = df['order_date'].dt.day_name()

# --- SIDEBAR ---
st.sidebar.title("🛒 Dashboard Navigation")
page = st.sidebar.radio("Choose a Page", ["🏠 Home", "📊 Exploratory Data Analysis", "🔮 Predict Returns"])
st.sidebar.markdown("---")

st.sidebar.header("Data Filters")
region_filter = st.sidebar.multiselect(
    "Select Region:",
    options=df["region"].unique(),
    default=df["region"].unique()
)
category_filter = st.sidebar.multiselect(
    "Select Category:",
    options=df["category"].unique(),
    default=df["category"].unique()
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This app analyzes e-commerce sales data and predicts customer returns "
    "based on order details. It's built from a Jupyter Notebook analysis."
)

# --- FILTER DATAFRAME BASED ON SIDEBAR SELECTION ---
df_selection = df.query(
    "region == @region_filter & category == @category_filter"
)

if 'order_month' not in df_selection.columns:
    df_selection['order_month'] = df_selection['order_date'].dt.month
if 'order_day_of_week' not in df_selection.columns:
    df_selection['order_day_of_week'] = df_selection['order_date'].dt.day_name()

# --- PAGE ROUTING ---
# --- HOME PAGE ---
if page == "🏠 Home":
    st.title("🏠 E-commerce Sales & Returns Dashboard")
    st.markdown("Welcome! This dashboard provides insights into sales data and a tool to predict order returns.")

    total_sales = int(df_selection["total_amount"].sum())
    avg_return_rate = round(df_selection["returned_numeric"].mean() * 100, 2)
    avg_profit_margin = round(df_selection["profit_margin"].mean(), 2)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Total Sales 💰", value=f"${total_sales:,}")
    kpi2.metric(label="Average Return Rate 🔄", value=f"{avg_return_rate}%")
    kpi3.metric(label="Average Profit Margin 📈", value=f"{avg_profit_margin}%")

    st.markdown("---")
    st.header("Dataset Preview")
    st.dataframe(df_selection.head(20))

# --- EXPLORATORY DATA ANALYSIS PAGE ---
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Interactive Exploratory Data Analysis")
    st.markdown("Dive deep into the data with these interactive charts. Use the sidebar filters to customize your view.")

    # Custom color palettes for variety
    palette1 = ["#6c63ff", "#48c9b0", "#f7cac9", "#ffe156", "#ff6f61"]
    palette2 = ["#2e86de", "#00b894", "#fdcb6e", "#d35400", "#636e72"]
    palette3 = ["#a29bfe", "#00b894", "#fd79a8", "#fab1a0", "#55efc4"]
    palette4 = ["#00b894", "#fd79a8", "#636e72", "#ffe156", "#6c63ff"]
    palette5 = ["#fab1a0", "#2e86de", "#ffe156", "#636e72", "#ff6f61"]

    col1, col2 = st.columns(2)
    with col1:
        # 1. Average Profit Margin by Category
        category_profit = df_selection.groupby('category')['profit_margin'].mean().sort_values().reset_index()
        fig1 = px.bar(category_profit, x='profit_margin', y='category', title='Average Profit Margin by Category',
                      orientation='h', color='category', color_discrete_sequence=palette1)
        fig1.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            xaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            yaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Orders by Payment Method
        payment_counts = df_selection['payment_method'].value_counts().reset_index()
        payment_counts.columns = ['payment_method', 'count']
        fig2 = px.pie(payment_counts, names='payment_method', values='count', title='Orders by Payment Method',
                      color_discrete_sequence=palette2, hole=0.4)
        fig2.update_traces(textinfo='percent+label')
        fig2.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Orders by Region
        region_counts = df_selection['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'count']
        fig3 = px.bar(region_counts, x='region', y='count', title='Orders by Region',
                      color='region', color_discrete_sequence=palette3)
        fig3.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            xaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            yaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig3, use_container_width=True)


        # 5. Average Delivery Time by Region
        avg_delivery = df_selection.groupby('region')['delivery_time_days'].mean().reset_index()
        fig5 = px.bar(avg_delivery, x='region', y='delivery_time_days', title='Average Delivery Time by Region',
                      color='region', color_discrete_sequence=palette5)
        fig5.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            xaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            yaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig5, use_container_width=True)

        # 6. Orders by Customer Gender
        gender_counts = df_selection['customer_gender'].value_counts().reset_index()
        gender_counts.columns = ['customer_gender', 'count']
        fig6 = px.pie(gender_counts, names='customer_gender', values='count', title='Orders by Customer Gender',
                      color_discrete_sequence=palette1, hole=0.3)
        fig6.update_traces(textinfo='percent+label')
        fig6.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig6, use_container_width=True)

        

        # 16. Delivery Time vs. Return Rate
        delivery_returns = df_selection.groupby('delivery_time_days')['returned_numeric'].mean().reset_index()
        delivery_returns['return_rate_pct'] = delivery_returns['returned_numeric'] * 100
        fig16 = px.area(delivery_returns, x='delivery_time_days', y='return_rate_pct',
                        title='Return Rate Increases with Delivery Time', markers=True, color_discrete_sequence=palette1)
        fig16.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig16, use_container_width=True)

        # 13. Return Rate by Payment Method
        payment_return = df_selection.groupby('payment_method')['returned_numeric'].mean().reset_index()
        payment_return['return_rate_pct'] = payment_return['returned_numeric'] * 100
        fig13 = px.bar(payment_return, x='payment_method', y='return_rate_pct', title='Return Rate by Payment Method',
                       color='payment_method', color_discrete_sequence=palette3)
        fig13.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            xaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            yaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig13, use_container_width=True)

    with col2:
        # 7. Return Rate by Region
        region_returns = df_selection.groupby('region')['returned_numeric'].mean().sort_values().reset_index()
        region_returns['return_rate_pct'] = region_returns['returned_numeric'] * 100
        fig7 = px.bar(region_returns, x='return_rate_pct', y='region', title='Return Rate by Region (%)',
                      orientation='h', color='region', color_discrete_sequence=palette2)
        fig7.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            xaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            yaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig7, use_container_width=True)

        # 8. Payment Method by Region
        fig8 = px.sunburst(df_selection, path=['region', 'payment_method'], title='Payment Method Distribution by Region',
                           color='region', color_discrete_sequence=palette3)
        fig8.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig8, use_container_width=True)

        # 9. Average Total Amount by Category
        avg_total = df_selection.groupby('category')['total_amount'].mean().reset_index()
        fig9 = px.bar(avg_total, x='category', y='total_amount', title='Average Total Amount by Category',
                      color='category', color_discrete_sequence=palette4)
        fig9.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            xaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            yaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig9, use_container_width=True)

        # 10. Orders by Month
        month_counts = df['order_month'].value_counts().sort_index().reset_index()
        month_counts.columns = ['order_month', 'count']
        fig10 = px.line(month_counts, x='order_month', y='count', title='Orders by Month',
                        markers=True, color_discrete_sequence=palette5)
        fig10.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig10, use_container_width=True)

        # 11. Orders by Day of Week
        day_counts = df_selection['order_day_of_week'].value_counts().sort_index().reset_index()
        day_counts.columns = ['order_day_of_week', 'count']
        fig11 = px.bar(day_counts, x='order_day_of_week', y='count', title='Orders by Day of Week',
                       color='order_day_of_week', color_discrete_sequence=palette1)
        fig11.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            xaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            yaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig11, use_container_width=True)

        # 12. Average Customer Age by Category
        avg_age = df_selection.groupby('category')['customer_age'].mean().reset_index()
        fig12 = px.scatter(avg_age, x='category', y='customer_age', title='Average Customer Age by Category',
                           color='category', color_discrete_sequence=palette2, size='customer_age')
        fig12.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig12, use_container_width=True)

        

        # 14. Return Rate by Category
        category_return = df_selection.groupby('category')['returned_numeric'].mean().reset_index()
        category_return['return_rate_pct'] = category_return['returned_numeric'] * 100
        fig14 = px.bar(category_return, x='category', y='return_rate_pct', title='Return Rate by Category',
                       color='category', color_discrete_sequence=palette4)
        fig14.update_layout(
            plot_bgcolor="#222", paper_bgcolor="#222",
            font=dict(color="#fff", size=16),
            title_font=dict(color="#fff", size=20),
            xaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            yaxis=dict(title_font=dict(color="#fff", size=16), tickfont=dict(color="#fff", size=14)),
            legend=dict(font=dict(color="#fff", size=14))
        )
        st.plotly_chart(fig14, use_container_width=True)


       


# --- PREDICT RETURNS PAGE ---
elif page == "🔮 Predict Returns":
    st.title("🔮 Predict Customer Returns")
    st.markdown("Enter the details of an order to get a prediction on whether it will be returned.")

    # --- NEW: Create a centered layout ---
    # We create 3 columns; the middle one is twice as wide as the sides.
    # The side columns act as empty spacers.
    left_spacer, main_col, right_spacer = st.columns([1, 2, 1])

    # --- Place all content inside the main middle column ---
    with main_col:
        with st.form("prediction_form"):
            st.header("Enter Order Details")
            # The 3 columns for the form are now *inside* the main centered column
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.selectbox("Product Category", options=sorted(df['category'].unique()))
                payment_method = st.selectbox("Payment Method", options=sorted(df['payment_method'].unique()))
                region = st.selectbox("Region", options=sorted(df['region'].unique()))
            with col2:
                total_amount = st.number_input("Total Amount ($)", min_value=0.0, step=10.0, value=100.0)
                shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, step=1.0, value=5.0)
                profit_margin = st.number_input("Profit Margin (%)", min_value=-100.0, max_value=100.0, step=1.0, value=20.0)
            with col3:
                delivery_time_days = st.slider("Delivery Time (Days)", min_value=1, max_value=30, value=5)
                customer_age = st.slider("Customer Age", min_value=18, max_value=80, value=35)
                customer_gender = st.radio("Customer Gender", options=sorted(df['customer_gender'].unique()))

            submit_button = st.form_submit_button(label='Get Prediction', use_container_width=True)

        if submit_button:
            input_data = pd.DataFrame({
                'category': [category], 'payment_method': [payment_method], 'delivery_time_days': [delivery_time_days],
                'region': [region], 'total_amount': [total_amount], 'shipping_cost': [shipping_cost],
                'profit_margin': [profit_margin], 'customer_age': [customer_age], 'customer_gender': [customer_gender]
            })

            prediction = model_pipeline.predict(input_data)
            prediction_proba = model_pipeline.predict_proba(input_data)

            st.header("Prediction Result")
            result_col, proba_col = st.columns([2, 1])
            with result_col:
                if prediction[0] == 'Yes':
                    st.error("Prediction: This order is LIKELY to be RETURNED.", icon="🚨")
                else:
                    st.success("Prediction: This order is UNLIKELY to be returned.", icon="✅")
            with proba_col:
                st.metric(label="Confidence Score", value=f"{prediction_proba.max()*100:.1f}%")

            st.markdown("---")
            st.header("🔍 What Influenced this Prediction?")

            importances = model_pipeline.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False).head(10)

            fig_importance = px.bar(
                feature_importance_df, x='importance', y='feature', orientation='h',
                title='Top 10 Factors Influencing the Prediction'
            )
            fig_importance.update_layout(
                plot_bgcolor="#222", paper_bgcolor="#222",
                font=dict(color="#fff", size=16),
                title_font=dict(color="#fff", size=20),
                yaxis={'categoryorder':'total ascending'}, yaxis_title=None, xaxis_title="Importance Score",
                legend=dict(font=dict(color="#fff", size=14))
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            st.info("Higher scores mean the model relied more heavily on that factor to make its decision.")
        
