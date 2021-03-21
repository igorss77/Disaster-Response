# Disaster-Response

This project seek to predict the message from disaster in multicategories.

# Files
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- emergency.db   # database to save clean data to

- motebooks
|- etl_pipeline.ipynb  # pipeline with etl  
|- ml_pipeline.ipynb  # pipeline with ml process

- models
 |- train_classifier.py
 |- classifier.pkl  # saved model
 ```
# Requirements

* scikit-learn==0.19.1 \
* pandas==0.23.3 \ 
* numpy==1.12.1 \
* nltk==3.2.5 \
* lightgbm==3.1.1
