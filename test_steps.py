import pytest
from pyspark.sql import SparkSession
from steps import read_raw, enrich_customers_products, enrich_orders, aggregate_profit


@pytest.fixture(scope="session")
def spark():
    s = SparkSession.builder.master("local[1]").appName("real-data-tests").getOrCreate()
    yield s
    s.stop()


def test_step1_read_raw_files(spark):
    folder = "/content"   
    customers, products, orders = read_raw(spark, folder)
    assert customers.count() > 0
    assert products.count() > 0
    assert orders.count() > 0
    assert "Customer ID" in customers.columns
    assert "Product ID" in products.columns
    assert "Order ID" in orders.columns


def test_step2_enriched_tables(spark):
    folder = "/content"
    customers, products, _ = read_raw(spark, folder)
    enriched_customers, enriched_products = enrich_customers_products(customers, products)
    assert "Customer Name" in enriched_customers.columns
    assert "Country" in enriched_customers.columns
    assert "Category" in enriched_products.columns
    assert "Sub-Category" in enriched_products.columns
    assert enriched_products.count() <= products.count()  


def test_step3_enriched_orders(spark):
    folder = "/content"
    customers, products, orders = read_raw(spark, folder)
    enriched_customers, enriched_products = enrich_customers_products(customers, products)
    enriched_orders = enrich_orders(orders, enriched_customers, enriched_products)
    assert "Profit" in enriched_orders.columns
    assert "Customer Name" in enriched_orders.columns
    assert "Category" in enriched_orders.columns
    assert enriched_orders.count() == orders.count()   


def test_step4_aggregate_profit(spark):
    folder = "/content"
    customers, products, orders = read_raw(spark, folder)
    enriched_customers, enriched_products = enrich_customers_products(customers, products)
    enriched_orders = enrich_orders(orders, enriched_customers, enriched_products)
    agg = aggregate_profit(enriched_orders)
    assert "Year" in agg.columns
    assert "Profit" in agg.columns
    #sanity check
    total_orders = enriched_orders.groupBy().sum("Profit").collect()[0][0]
    total_agg = agg.groupBy().sum("Profit").collect()[0][0]
    assert total_orders == pytest.approx(total_agg)


def test_pipeline_integration(spark):
    folder = "/content"
    customers, products, orders = read_raw(spark, folder)
    enriched_customers, enriched_products = enrich_customers_products(customers, products)
    enriched_orders = enrich_orders(orders, enriched_customers, enriched_products)
    agg = aggregate_profit(enriched_orders)
    # pipeline sanity checks
    assert customers.count() > 0
    assert products.count() > 0
    assert orders.count() > 0
    assert enriched_orders.count() == orders.count()
    assert agg.count() > 0
