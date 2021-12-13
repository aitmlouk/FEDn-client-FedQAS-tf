# FedQAS  project
Machine reading comprehension (MRC) of text data is one important task in Natural Language Understanding. It is a complex NLP problem with a lot of ongoing research since the release of the Stanford Question Answering Dataset (SQuAD) and CoQA. It is considered as an effort to teach computers how to "understand" a text, and then to be able to answer questions from it using deep learning. However, until now large-scale training on private text data and knowledge sharing has been missing for this NLP task.
In this project, we implemented FedQAS, a privacy-preserving machine reading system that leverages large-scale private data. The implementation combines Transformer models and Federated learning technologies using FEDn framework. The proposed approach can be useful for industries seeking similar solutions and especially where the data are private and cannot be shared. This implementation is inspired from keras.io

## Configure and start a client using cpu device
The easiest way to start clients for quick testing is to use shell script.The following 
shell script will configure and start a client on a blank Ubuntu 20.04 LTS VM:    


```bash
#!/bin/bash

# Install Docker and docker-compose
sudo apt-get update
sudo sudo snap install docker

# clone the fedqas-keras example
git clone https://github.com/aitmlouk/FEDn-client-fedqas-keras.git
cd FEDn-client-fedqas-keras

# if no available data, download it from archive
# wget https://archive.org/download/data_20211213/data.zip
# sudo apt install unzip
# unzip data.zip
# sudo rm data.zip

# Make sure you have edited extra-hosts.yaml to provide hostname mappings for combiners
# Make sure you have edited fedn-network.yaml to provide hostname mappings for reducer
sudo docker-compose -f docker-compose.yaml -f extra-hosts.yaml up --build
```

### Start prediction- Global model serving
We have made it possible to use the trained global model for testing, prediction and annotation, to start the UI make sure that the base_services (fedn/config) is
is started and run the flask app (python prediction/app.py)
```bash
# prediction/
python app.py
```

### Configuring the tests
We have made it possible to configure a couple of settings to vary the conditions for the training. These configurations are exposed in the file 'client/settings.yaml': 

```yaml 
# Parameters for the model and local training
max_seq_length: 384
learning_rate: 5e-5
batch_size: 8
epochs: 1 # For demonstration, 3 epochs are recommended
verbose: True
```

### Creating a compute package
To train a model in FEDn you provide the client code (in 'client') as a tarball. For convenience, we ship a pre-made package (nlp_imdb.tar.gz). Whenever you make updates to the client code (such as altering any of the settings in the above mentioned file), you need to re-package the code (as a .tar.gz archive) and copy the updated package to 'packages'.

```bash
tar -czvf package.tar.gz client
```

## Creating a seed model
The model architecture is specified in the file 'client/init_model.py'. This script creates an untrained neural network and serialized that to a file, which is uploaded as the initial model for Federated training. For convenience we ship a pregenerated initial model in the 'initial_model/' directory. If you wish to alter the base model, edit 'client/models/squad_model.py' and regenerate the seed file:
```bash
# client/models
python init_model.py 
```

## License
Apache-2.0 (see LICENSE file for full information).