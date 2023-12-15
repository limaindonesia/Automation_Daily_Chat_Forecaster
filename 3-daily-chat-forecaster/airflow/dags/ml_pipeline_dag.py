from datetime import datetime

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.edgemodifier import Label

@dag(
    dag_id='ml_pipeline',
    schedule_interval='@weekly',
    start_date=datetime(2023, 12, 11),
    catchup=False,
    tags=["feature-engineering", "model-training", "batch-prediction"],
    max_active_runs=1
)

def ml_pipeline():
    @task.virtualenv(
            task_id="run_feature_pipeline",
            requirements=[],
            python_version="3.9",
            multiple_outputs=True,
            system_site_packages=True,
    )
    def run_feature_pipeline(
        date_start: str,
        date_end: str
    ):

        from pipeline import utils, feature_pipeline

        logger = utils.get_logger(__name__)

        logger.info(f"date_start = {date_start}")
        logger.info(f"date_end = {date_end}")

        return feature_pipeline.run(
            date_start=date_start,
            date_end=date_end
        )

    @task.virtualenv(
            task_id="run_training_pipeline",
            requirements=[],
            python_version="3.9",
            multiple_outputs=True,
            system_site_packages=True,
    )
    def run_training_pipeline():
        
        from pipeline import utils, training_pipeline

        logger = utils.get_logger(__name__)

        return training_pipeline.run() 

    @task.virtualenv(
            task_id="run_training_pipeline",
            requirements=[],
            python_version="3.9",
            multiple_outputs=True,
            system_site_packages=True,
    )
    def run_inference_pipeline():

        from pipeline import utils, inference_pipeline

        logger = utils.get_logger(__name__)

        return inference_pipeline.run()
    
    # Define Airflow variables
    date_start = str(Variable.get("ml_pipeline_date_start", default_var='2023-05-06'))
    date_end = str(Variable.get("ml_pipeline_date_end", default_var='2023-11-13'))

    # Feature pipeline
    feature_pipeline_step = run_feature_pipeline(
        date_start=date_start,
        date_end=date_end
    )
    # Training pipeline
    training_pipeline_step = run_training_pipeline()
    # Inference pipeline
    batch_predict_step = run_inference_pipeline()
    # Define DAG Structure
    (
        feature_pipeline_step
        >> training_pipeline_step
        >> batch_predict_step
    )

ml_pipeline()