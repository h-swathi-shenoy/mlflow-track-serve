import mlflow

mlflow_run_id = 'f8a2626f0e93463db6c3527793d961c3'

model_name = 'cat_vs_dog_classifier'
version = 1
logged_model_path = f"runs:/{mlflow_run_id}/models"


with mlflow.start_run(run_id= mlflow_run_id) as mlrun:
    result = mlflow.register_model(logged_model_path, model_name)
    print(result)