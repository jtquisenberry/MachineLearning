import pandas as pd

def load_yelp_orig_data():
    PATH_TO_YELP_REVIEWS = INPUT_FOLDER + '/review.json'

    # read the entire file into a python array
    with open(PATH_TO_YELP_REVIEWS, 'r') as f:
        data = f.readlines()

    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"

    # now, load it into pandas
    data_df = pd.read_json(data_json_str)

    data_df.head(100000).to_csv(OUTPUT_FOLDER + '/output_reviews_top.csv')

load_yelp_orig_data()