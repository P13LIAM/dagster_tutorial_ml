import csv
from datetime import datetime

import requests

from dagster import get_dagster_logger, job, op, repository, schedule


@op
def hello_cereal(context):
    response = requests.get("https://docs.dagster.io/assets/cereal.csv")
    lines = response.text.split("\n")
    cereals = [row for row in csv.DictReader(lines)]
    date = context.op_config["date"]
    get_dagster_logger().info(f"Today is {date}. Found {len(cereals)} cereals.")


@job
def hello_cereal_job():
    hello_cereal()

@schedule(
    cron_schedule="5 15 * * *",
    job=hello_cereal_job,
    execution_timezone="Asia/Bangkok",
)
def good_morning_schedule(context):
    date = context.scheduled_execution_time.strftime("%Y-%m-%d")
    return {"ops": {"hello_cereal": {"config": {"date": date}}}}

@repository
def hello_cereal_repository():
    return [hello_cereal_job, good_morning_schedule]