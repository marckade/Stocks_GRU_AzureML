import azureml.core
from azureml.core import Workspace, Datastore
import mlflow

# Internal Functions
import ConnectionFunctions
import DataScripts.Transformation
import TrainScripts.TrainGRU

def main():
    workspace = ConnectionFunctions.connectAzureWorkspace()
    print(azureml.core.VERSION)
    # Create Experiment
    experiment_name = 'Adv_AI_LocalTrain'
    experiment = ConnectionFunctions.createExperiment(experiment_name, workspace)
    env = ConnectionFunctions.createEnvironment('user-managed-env')

    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
    print(workspace.get_mlflow_tracking_uri())
    mlflow.set_experiment(experiment_name)
    # mlflow.autolog()
    mlflow.tensorflow.autolog()

    print(mlflow.active_run)
    X_train, y_train, X_valid, y_valid, X_test = DataScripts.Transformation.transformData()

    GRUModel, predicted_stock_price = TrainScripts.TrainGRU.trainModel(X_train, y_train, X_valid, y_valid, X_test)

    print(type(GRUModel))
    print(GRUModel)
    # mlflow.run('TrainScripts/SubmitTrain.py', experiment_name = 'Adv_AI_LocalTrain')
    with mlflow.start_run() as run:
        # runfunc = ConnectionFunctions.createRun("/TrainScripts","SubmitTrain.py", env, experiment)
        mlflow.log_param("Hello Param", "World")
        mlflow.log_param("Predicted Prices", 3)
        # mlflow.tensorflow.log_model(GRUModel, 'Model')
        mlflow.sklearn.log_model(GRUModel, 'Model')

    print(run)
    print("Finished with Run")

if __name__ == "__main__":
    main()