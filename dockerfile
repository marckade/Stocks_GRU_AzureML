FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20220902.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/sklearn-0.24.1
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.7 pip=20.2.4

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'matplotlib>=3.3,<3.4' \
                'psutil~=5.8,<5.9' \
                'tqdm>=4.59,<4.60' \
                'pandas~=1.1,<1.2' \
                'scipy~=1.5,<1.6' \
                'numpy' \
                'ipykernel~=6.0' \
                'azureml-core==1.45.0' \
                'azureml-defaults==1.45.0' \
                'azureml-mlflow==1.45.0' \
                'azureml-telemetry==1.45.0' \
                'tensorflow' \
                'scikit-learn~=0.24.1'

# RUN export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python                          
# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

COPY . .