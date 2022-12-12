from azureml.core import Workspace
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Experiment
import json

def connectAzureWorkspace():
    with open('config.json', 'r') as json_files:
        json_load = json.load(json_files)

    subscription_id = json_load['subscription_id']
    resource_group = json_load['resource_group']
    workspace_name = json_load['workspace_name']

    try:
        ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
        ws.write_config()
        print('Library configuration success')
        return ws
    except:
        print('Workspace not found')


def createExperiment(experimentName, workspace):
    experiment_name = experimentName
    exp = Experiment(workspace = workspace, name = experiment_name)
    return exp

def createEnvironment(envString):
    user_managed_env = Environment(envString)
    user_managed_env.python.user_managed_dependencies = True

    return user_managed_env

def createRun(source_directory_name, script_name, environment, experiment):
    src = ScriptRunConfig(source_directory = source_directory_name, script = script_name, environment = environment)
    run = experiment.submit(src)
    return run
