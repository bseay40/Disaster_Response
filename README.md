# Disaster Response Pipeline Project
>  Project include a web app where an emergency worker can input a new message
>  and get classification results in several categories. The web app will also
>  display visualizations of the data.

# Installation needed
> Run on Python 3.8.5. Packages used outside of those included in the Anaconda distribution are listed here:
>> sqlalchemy
>> nltk
>> sklearn

# Files for project
> User inputs both the message and category datasets.

# How to interact with the project
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. In a separate terminal type.
    `env|grep WORK`

4. In a new browser window, type in the following:
    `https://SPACEID-3001.SPACEDOMAIN`
   where SPACEID and SPACEDOMAIN are provided in step 3.

# Licensing, Authors, Acknowledgements
> Thanks Udacity.
