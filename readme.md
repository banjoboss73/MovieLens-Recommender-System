### Ensure that the data files are downloaded
In order to run the program, you need to ensure that the relevant data files are downloaded. 

This includes:
 - The Movielens 100k (`ml-100k.zip`) data set: https://grouplens.org/datasets/movielens/
 - The US county zipcodes dataset: https://www.kaggle.com/datasets/danofer/zipcodes-county-fips-crosswalk

Unzip the files and add them to your folder

### Import the necessary packages
The details for all the packages you will need are in the `requirements.md` file

### Running the programme
You must run the programme through the command line using the following format:
`python3 run_eval.py --user_id <USER_ID> --num_recs <NUMBER_OF_RECOMMENDATIONS> --eval|--basic | --advanced`

The user ID denotes which user is being selected for the recommendations.

####  Functions
The three optional commands are `--eval`, `--basic` and `advanced`. Only one of these can be chosen.
 - `--eval`: Evaluates the RMSE scores for the basic and advanced models
 - `--basic`: Generates recommendations for the basic model along with the average novelty score for the inputted user.
 - `--advanced`: Generates recommendations for the advanced model along with the average novelty score for the inputted user.

All three inputs will need to written, even for the evaluation function.

#### Error messages
When using the command line, please ensure:
 - The user ID is not empty.
 - The user ID is an integer value.

Error messages will appear if any of these criteria are not met and you can run the command line again.

