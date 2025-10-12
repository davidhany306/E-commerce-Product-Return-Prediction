import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import os
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="E-commerce Returns Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- HELPER FUNCTIONS ----------------
@st.cache_data
def load_data():
    """Loads the cleaned dataset."""
    df = pd.read_csv("cleaned_final_project.csv")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.rename(columns={'returned?': 'returned'}, inplace=True)
    df['returned_numeric'] = df['returned'].astype(str).apply(lambda x: 1 if x.strip().lower() == 'yes' else 0)
    return df

@st.cache_resource
def load_final_pipeline():
    model = joblib.load("ocelot.pkl")
    try:
        feature_names = model.named_steps['Preprocessing'].get_feature_names_out()
    except:
        feature_names = []
    return model, feature_names

def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(30, 30, 30, 0.9) !important;
        color: white;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOAD DATA & MODEL ----------------
if os.path.exists("background.jpeg"):
    set_bg("background.jpeg")

df = load_data()
model_pipeline, feature_names = load_final_pipeline()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛒 Dashboard Navigation")
page = st.sidebar.radio("Choose a Page", ["🏠 Home", "📊 Exploratory Data Analysis", "🔮 Predict Returns"])
st.sidebar.markdown("---")

st.sidebar.header("Data Filters")
region_filter = st.sidebar.multiselect("Select Region:", options=df["region"].unique(), default=df["region"].unique())
category_filter = st.sidebar.multiselect("Select Category:", options=df["category"].unique(), default=df["category"].unique())
df_selection = df.query("region == @region_filter & category == @category_filter")
# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.title("🏠 E-commerce Sales & Returns Dashboard")
    st.markdown("This dashboard provides insights into sales data and a return prediction tool.")

    total_sales = int(df_selection["total_amount"].sum())
    avg_return_rate = round(df_selection["returned_numeric"].mean() * 100, 2)
    avg_profit_margin = round(df_selection["profit_margin"].mean(), 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales 💰", f"${total_sales:,}")
    c2.metric("Avg Return Rate 🔄", f"{avg_return_rate}%")
    c3.metric("Avg Profit Margin 📈", f"{avg_profit_margin}%")

    st.markdown("---")
    st.dataframe(df_selection.head(20))

# ---------------- EDA PAGE ----------------
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
# ---------------- PREDICT RETURNS PAGE ----------------
elif page == "🔮 Predict Returns":
    st.title("🔮 Predict Customer Returns")
    st.markdown("Enter order details to check if it’s likely to be returned.")

    left, main, right = st.columns([1, 2, 1])
    with main:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                category = st.selectbox("Category", sorted(df['category'].unique()))
                region = st.selectbox("Region", sorted(df['region'].unique()))
                order_month = st.selectbox("Order Month", sorted(df['order_month'].unique()))
            with c2:
                total = st.number_input("Total Amount ($)", 0.0, step=10.0, value=100.0)
                shipping = st.number_input("Shipping Cost ($)", 0.0, step=1.0, value=5.0)
                profit = st.number_input("Profit Margin (%)", -100.0, 100.0, 20.0)
                order_day = st.slider("Order Day", 1, 31, 15)
            with c3:
                delivery = st.slider("Delivery Time (Days)", 1, 30, 5)
                age = st.slider("Customer Age", 18, 80, 35)
                gender = st.radio("Gender", sorted(df['customer_gender'].unique()))
                payment = st.selectbox("Payment Method", sorted(df['payment_method'].unique()))
                order_year = st.selectbox("Order Year", sorted(df['order_year'].unique()))

            submitted = st.form_submit_button("🧠 Get Prediction", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame({
                'category': [category],
                'payment_method': [payment],
                'delivery_time_days': [delivery],
                'region': [region],
                'total_amount': [total],
                'shipping_cost': [shipping],
                'profit_margin': [profit],
                'customer_age': [age],
                'customer_gender': [gender],
                'price': [df['price'].mean()],
                'quantity': [1],
                'discount': [df['discount'].mean()],
                'order_day': [order_day],
                'order_month': [order_month],
                'order_year': [order_year],
                'order_day_of_week': [0]
            })

            dow_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                       'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            input_df['order_day_of_week'] = input_df['order_day_of_week'].map(dow_map).fillna(0)

            st.expander("🔍 Model Input Preview").markdown(input_df.to_markdown())

            pred = model_pipeline.predict(input_df)
            prob = model_pipeline.predict_proba(input_df)

            if pred[0] == 'Yes':
                st.error("🚨 Likely to be RETURNED")
            else:
                st.success("✅ Unlikely to be Returned")

            st.metric("Confidence", f"{prob.max()*100:.1f}%")

            # Feature importance
            st.markdown("---")
            st.subheader("📊 Top Features")
            try:
                importances = model_pipeline.named_steps['Model'].feature_importances_
                feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feat_df = feat_df.sort_values(by='importance', ascending=False).head(10)
                fig_imp = px.bar(feat_df, x='importance', y='feature', orientation='h', title='Feature Importances')
                st.plotly_chart(fig_imp, use_container_width=True)
            except:
                st.info("Feature importances not available for this model.")
