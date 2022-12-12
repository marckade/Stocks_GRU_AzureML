import azureml.core
from azureml.core import Workspace, Datastore
import mlflow

# Internal Functions
import ConnectionFunctions
import DataScripts.Transformation
import TrainScripts.TrainGRU
def main():
    workspace = ConnectionFunctions.connectAzureWorkspace()

    # Create Experiment
    experiment_name = 'Adv_AI_LocalTrain'
    experiment = ConnectionFunctions.createExperiment(experiment_name, workspace)
    env = ConnectionFunctions.createEnvironment('user-managed-env')

    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()

    X_train, y_train, X_valid, y_valid, X_test = DataScripts.Transformation.transformData()

    TrainScripts.TrainGRU.trainModel(X_train, y_train, X_valid, y_valid, X_test)

    
if __name__ == "__main__":
    main()