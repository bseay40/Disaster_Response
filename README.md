# Disaster Response Pipeline Project
> Summary of the project

# Installation needed
> Run on Python 3.8.5. Packages used outside of those included in the Anaconda distribution are listed here:
>>
>>
>>

# Project motivation(s)
>

# Files for project
>

# How to interact with the project
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Licensing, Authors, Acknowledgements
>
