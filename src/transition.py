import mlflow


if __name__=='__main__':
    model_name = 'cat_vs_dog_classifier'
    model_version = 1

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage='Production'
    )