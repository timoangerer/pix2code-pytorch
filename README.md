# PIX2CODE Pytorch 1.8 implementation

# Setup and run model
Follow the follwing steps to train and evaluate the model localy on your machine.

1. Clone the repository

2. Install the depdendecies  
    This project was written with python version `3.8.8`.

    Run the following command to create a conda environment named `pix2code` with the requried dependencies:

        conda env create python=3.8.8 -f environment.yml

    or install all the dependencies manually:

        conda install -y -c pytorch pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 
        conda install -y -c conda-forge tqdm=4.60.0 pillow=8.2.0 nltk=3.6.1
        conda install -y -c conda-forge nb_conda_kernels=2.3.1 jupyterlab=3.0.12

3. Download the dataset

    Create a new folder "data" inside the project root folder, download the dataset and extract it into the data folder.
    
    The datasets that was used is the **pix2code** dataset. Download the dataset from one of the following links: [Google drive](https://drive.google.com/u/1/uc?id=1OKQZJUP6Xml_opVsR97H0H0kKfgmOxcF&export=download), [GitHub](https://github.com/tonybeltramelli/pix2code/tree/master/datasets).

    This is what the folder structure of the `data` folder should look like:

        data
        ├── ...
        └── web
            └── all_data
                ├── AF4840B2-2B9F-4ED0-A58D-E260B14858E1.gui
                └── ...

    The pix2code dataset contains three sub-datasets. The following steps are only concered with the `web` dataset.

4. Split the dataset

    The `train.py` and `evaluate.py` scripts assume the existance of three datasplit files `train_dataset.txt`, `test_dataset.txt` and `validation_dataset.txt`, each containing the IDs of the data examples for the respective data split. The datasplit files have to be at the same folder level as the folder containing the data examples.

    Run `split_data.py` to generate the data split files for the `web` dataset:

        python split_data.py --data_path=./data/web/all_data

    This is what the `data` folder should look like up to this point:

        data
        ├── ...
        └── web
            ├── all_data
            |   ├── AF4840B2-2B9F-4ED0-A58D-E260B14858E1.gui
            │   └── ...
            ├── test_dataset.txt
            ├── train_dataset.txt
            └── validation_dataset.txt

5. Create the vocabulary file

    You need to generate a `vocab.txt` file that contains all the tokens the model should be able to predict, separeted by whitespaces.

    Run `build_vocab.py` to generate a vocabulary file based on the tokens that appear in the specified dataset.

        python `build_vocab.py` --data_path=./data/web/all_data

    This is what the `data` folder should look like up to this point:

        data
        ├── ...
        └── web
            ├── all_data
            |   ├── AF4840B2-2B9F-4ED0-A58D-E260B14858E1.gui
            │   └── ...
            ├── test_dataset.txt
            ├── train_dataset.txt
            ├── validation_dataset.txt
            └── vocab.txt

6. Train the model

    Run the following command to train the model:

        python train.py --data_path=./data/web/all_data --epochs=15 --save_after_epochs=5 --batch_size=4 --split=train

7. Evaluate the model

    Run the following command to evaluate the model:

        python evaluate.py --data_path=./data/web/all_data --model_file_path=<path-to-model-file> --split=validation --viz

    To visualize the results of the model evaluation, run the evaluation script with the `--viz` flag and then follow the steps inside the `visualize_inference.ipynb`.

    