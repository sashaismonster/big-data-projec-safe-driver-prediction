**Porto Seguro's Safe Driver Prediction**
This repository contains the final project for the Big Data course at IU-2023. The project focuses on the Porto Seguro's Safe Driver Prediction dataset, a real-world data science problem.

**Repository Structure**
- data/: Contains the dataset files.
- models/: Contains the Spark ML models.
- notebooks/: Contains Zeppelin notebooks used for learning and exploration purposes.
- output/: Stores the results of the project, including CSV files, text files, images, and other output materials.
- scripts/: Stores .sh and .py scripts for the pipeline stages.
- sql/: Stores all .sql and .hql files.
- requirements.txt: Lists the Python packages needed for running the Python scripts. Feel free to add more packages when necessary.
- main.sh: The main script that runs all scripts of the pipeline stages, executing the full pipeline and storing the results in the output/ folder.

**Dataset Information**
The dataset used in this project is the Porto Seguro's Safe Driver Prediction dataset. The goal is to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. A more accurate prediction will allow insurance companies to further tailor their prices, making auto insurance coverage more accessible to more drivers.

The dataset is provided by Porto Seguro, one of Brazil's largest auto and homeowner insurance companies. The dataset is imbalanced, with most insurance policy holders not filing a claim. The dataset consists of 59 variables, including 'ID' and 'Target', with a total of 57 features used for prediction. All of the 57 features are anonymized for privacy reasons.

You can find more information about the dataset and the competition on the Kaggle competition page.
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

**Usage**
To run the project, execute the main.sh script. This will run all scripts of the pipeline stages, executing the full pipeline and storing the results in the output/ folder.

`./main.sh`

**Note**
The notebooks in the notebooks/ folder are used only for learning purposes. All Python scripts of the pipeline are located in the scripts/ folder. During the assessment, the notebooks/ folder can be deleted to check that the pipeline does not depend on its content.

**Requirements**
Python packages needed for running the Python scripts are listed in the requirements.txt file. Install the required packages with:

`pip install -r requirements.txt`

**License**
This project is licensed under the terms of the MIT license.
