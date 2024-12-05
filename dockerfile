# Use the Miniconda base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the environment.yml file into the container
COPY environment.yml .

# Create the Conda environment from the yml file
RUN conda env create -f environment.yml

# Activate the environment by modifying the PATH
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Set the default command to run when starting the container
CMD ["bash"]