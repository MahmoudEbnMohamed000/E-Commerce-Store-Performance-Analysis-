import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"D:\Data Science Truck\archive (1)\ecommerce_sales_analysis.csv")
# Explore the data
print(df.head(10))
print(df.tail(10))
print(df.shape)
print(df.info())
print(df.describe())
print(df.isna().sum())
sns.heatmap(df.isnull())
plt.show()
print(df.duplicated().sum())

#Analysis of the data
#(Books) ما هو المنتج الاكثر مبيعا ؟  
#الأعمدة التي تحتوي على مبيعات كل شهر
sales_cols = [
    col for col in df.columns
    if col.startswith("sales_month_")
]

# التأكد من أن قيم المبيعات أرقام
df[sales_cols] = df[sales_cols].apply(pd.to_numeric, errors="coerce")

# إجمالي الوحدات المباعة لكل صف
df["total_units_sold"] = df[sales_cols].sum(axis=1)

# تجميع المبيعات حسب المنتج، ثم ترتيب المنتجات من الأكثر مبيعًا
product_sales = (
    df.groupby(["product_id", "product_name"], as_index=False)
      ["total_units_sold"].sum()
      .sort_values("total_units_sold", ascending=False)
)

# عرض المنتج الأكثر مبيعًا 
best_product = product_sales.iloc[0]
print(
    f"best seller: {best_product['product_name']} "
    f"number {best_product['total_units_sold']} وحدة"
)

print(product_sales.head(1))

# رسم أعلى 1 منتجات
top_1 = product_sales.head(1).sort_values("total_units_sold")

top_1.plot(
    kind="barh",
    x="product_name",
    y="total_units_sold",
    legend=False
)
plt.xlabel("Total units sold")
plt.ylabel("Product")
plt.title("Top 1 Best-Selling Products")
plt.tight_layout()
plt.show()


#ما أكثر الشهور تحقيقًا للإيرادات؟
#revenue = sales * price 
monthly_revenue = {
    f"Month {month}": (df[f"sales_month_{month}"] * df["price"]).sum()
    for month in range(1, 13)
}

monthly_revenue = pd.Series(monthly_revenue, name="revenue").sort_values(
    ascending=False
)

print("\n monthly_revenue:")
print(monthly_revenue)
print(
    f"\ the best monthly revenue : {monthly_revenue.index[0]} "
    f"value {monthly_revenue.iloc[0]:,.2f}"
)

monthly_revenue.sort_values().plot(kind="barh", color="green")
plt.xlabel("Revenue")
plt.ylabel("Month")
plt.title("Revenue by Month")
plt.tight_layout()
plt.show()

#ما العملاء أو المناطق التي تحقق أعلى قيمة؟
# لا تحتوي الداتا الحالية على أعمدة للعملاء أو المناطق.
# البديل المتاح: مقارنة إجمالي الإيرادات حسب الفئة.
sales_cols = [f"sales_month_{month}" for month in range(1, 13)]
df["total_revenue"] = df[sales_cols].mul(df["price"], axis=0).sum(axis=1)

category_revenue = df.groupby("category")["total_revenue"].sum().sort_values(
    ascending=False
)

print("\  revenue by category:")
print(category_revenue)
print(
    f"\ the best revenue category : {category_revenue.index[0]} "
    f" value {category_revenue.iloc[0]:,.2f}"
)

category_revenue.sort_values().plot(kind="barh", color="steelblue")
plt.xlabel("Revenue")
plt.ylabel("Category")
plt.title("Revenue by Category")
plt.tight_layout()
plt.show()

#لماذا قد تنخفض المبيعات؟
monthly_units = pd.Series(
    {
        f"Month {month}": df[f"sales_month_{month}"].sum()
        for month in range(1, 13)
    },
    name="units_sold",
)
monthly_change = monthly_units.diff()

print("\ Change in the number of units compared to the previous month:")
print(pd.DataFrame({"units_sold": monthly_units, "change": monthly_change}))

declines = monthly_change[monthly_change < 0]
if declines.empty:
    print("There is no month-over-month decrease in the total number of units sold.")
else:
    decline_month = declines.idxmin()
    month_number = int(decline_month.split()[-1])
    previous_column = f"sales_month_{month_number - 1}"
    decline_column = f"sales_month_{month_number}"

    category_changes = df.groupby("category")[[previous_column, decline_column]].sum()
    category_changes["change"] = (
        category_changes[decline_column] - category_changes[previous_column]
    )

    print(f"\n The biggest drop occurred in {decline_month}:")
    print(category_changes.sort_values("change").head())

df["annual_units_sold"] = df[sales_cols].sum(axis=1)
correlations = (
    df[["annual_units_sold", "price", "review_score", "review_count"]]
    .corr()["annual_units_sold"]
    .drop("annual_units_sold")
    .sort_values()
)
print("\n Correlation between annual sales and product characteristics (does not prove causality):")
print(correlations)




# Professional KPI Dashboard
# This dataset contains monthly sales and product-level attributes,
# but it does not contain an explicit order table or customer retention history.
# Therefore, some KPIs are calculated directly when possible and marked as unavailable
# when the required fields do not exist.

sales_cols = [f"sales_month_{month}" for month in range(1, 13)]

df = df.copy()
df[sales_cols] = df[sales_cols].apply(pd.to_numeric, errors="coerce")
df["annual_units_sold"] = df[sales_cols].sum(axis=1)
df["total_revenue"] = df[sales_cols].mul(df["price"], axis=0).sum(axis=1)

# 1) Total Revenue
# Formula: sum of (monthly units sold × product price)
total_revenue = df["total_revenue"].sum()

# 2) Total Orders
# If an order_id column exists, use unique orders.
# Otherwise, the dataset does not support a real order count.
if "order_id" in df.columns:
    total_orders = df["order_id"].nunique()
elif "order_number" in df.columns:
    total_orders = df["order_number"].nunique()
else:
    total_orders = None

# 3) Average Order Value (AOV)
# Formula: Total Revenue / Total Orders
if total_orders is not None and total_orders > 0:
    average_order_value = total_revenue / total_orders
else:
    average_order_value = None

# 4) Best-Selling Products
# Formula: total units sold by product_name / product_id
product_sales = (
    df.groupby(["product_id", "product_name"], as_index=False)["annual_units_sold"]
      .sum()
      .sort_values("annual_units_sold", ascending=False)
      .reset_index(drop=True)
)

best_selling_products = product_sales.head(10)

# 5) Monthly Revenue
monthly_revenue = pd.Series(
    {
        f"Month {month}": (df[f"sales_month_{month}"] * df["price"]).sum()
        for month in range(1, 13)
    },
    name="revenue",
).sort_values(ascending=False)

# 6) Customer Retention (if available)
# Logic: if the data contains customer_id and a purchase date, we can compute repeat-rate.
# This dataset does not contain customer-level purchase history.
if "customer_id" in df.columns and "purchase_date" in df.columns:
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    customers = df.groupby("customer_id").agg(
        first_purchase=("purchase_date", "min"),
        total_orders=("customer_id", "count"),
    )
    customers["is_repeat_customer"] = customers["total_orders"] > 1
    retention_rate = customers["is_repeat_customer"].mean() * 100
else:
    retention_rate = None

# Print KPI Dashboard
print("\n===== KPI Dashboard =====")
print(f"Total Revenue: {total_revenue:,.2f}")
if total_orders is not None:
    print(f"Total Orders: {total_orders:,.0f}")
else:
    print("Total Orders: Not available in the current dataset")
if average_order_value is not None:
    print(f"Average Order Value (AOV): {average_order_value:,.2f}")
else:
    print("Average Order Value (AOV): Not available because order-level data is missing")

print("\nBest-Selling Products:")
print(best_selling_products.head(10).to_string(index=False))

print("\nMonthly Revenue:")
print(monthly_revenue.to_string())

if retention_rate is not None:
    print(f"\nCustomer Retention Rate: {retention_rate:.2f}%")
else:
    print("\nCustomer Retention Rate: Not available because customer-level historical data is missing")

# Optional business summary
print("\nBusiness Insight Summary:")
print(f"- Highest monthly revenue: {monthly_revenue.index[0]} ({monthly_revenue.iloc[0]:,.2f})")
print(f"- Best-selling product: {best_selling_products.iloc[0]['product_name']} ({best_selling_products.iloc[0]['annual_units_sold']:.0f} units)")

# Optional: category revenue
category_revenue = df.groupby("category")["total_revenue"].sum().sort_values(ascending=False)
print("\nRevenue by Category:")
print(category_revenue.to_string())

# --- KPI Visualization ---
# These charts support the KPI summary for the metrics that are available in the dataset.
plt.figure(figsize=(10, 5))
plt.bar(best_selling_products["product_name"].head(8),
        best_selling_products["annual_units_sold"].head(8),
        color="steelblue")
plt.title("Top 8 Best-Selling Products")
plt.xlabel("Product")
plt.ylabel("Units Sold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
monthly_revenue.sort_values().plot(kind="barh", color="darkgreen")
plt.title("Monthly Revenue")
plt.xlabel("Revenue")
plt.ylabel("Month")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
category_revenue.sort_values().plot(kind="barh", color="purple")
plt.title("Revenue by Category")
plt.xlabel("Revenue")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# Optional summary metric chart for available KPIs
available_kpis = {
    "Total Revenue": total_revenue,
}
if average_order_value is not None:
    available_kpis["Average Order Value"] = average_order_value

plt.figure(figsize=(8, 5))
plt.bar(available_kpis.keys(), available_kpis.values(), color=["#1f77b4", "#ff7f0e"][:len(available_kpis)])
plt.title("Available KPI Summary")
plt.ylabel("Value")
plt.xticks(rotation=20)
for key, value in available_kpis.items():
    plt.text(key, value * 0.98, f"{value:,.0f}", ha="center", va="top", fontsize=9)
plt.tight_layout()
plt.show()
