import pandas as pd
import glob, json, os
from pyspark.sql import SparkSession, functions as F

# Read raw files
def read_raw(spark, folder):
    customer_file = glob.glob(os.path.join(folder, "Customer*.xlsx"))[0]
    product_file = glob.glob(os.path.join(folder, "Products*.csv"))[0]
    order_file   = glob.glob(os.path.join(folder, "Orders*.json"))[0]

    customers = pd.read_excel(customer_file, dtype=str).fillna("")
    products  = pd.read_csv(product_file)
    with open(order_file) as f:
        orders = pd.DataFrame(json.load(f))

    return (spark.createDataFrame(customers),
            spark.createDataFrame(products),
            spark.createDataFrame(orders))

# Clean
def enrich_customers_products(customers, products):
    customers = customers.select("Customer ID", "Customer Name", "Country")
    products  = products.dropDuplicates(["Product ID"]).select("Product ID", "Category", "Sub-Category")
    return customers, products

# Join
def enrich_orders(orders, customers, products):
    return (orders
        .withColumn("Profit", F.round("Profit", 2))
        .join(customers, "Customer ID", "left")
        .join(products, "Product ID", "left"))

# Agg
def aggregate_profit(orders):
    return (orders
        .withColumn("Year", F.year(F.to_date("Order Date", "d/M/yyyy")))
        .fillna("Unknown", subset=["Category", "Sub-Category"])
        .groupBy("Year", "Category", "Sub-Category", "Customer ID")
        .agg(F.round(F.sum("Profit"), 2).alias("Profit")))
