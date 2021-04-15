# import libraries
import os

import pandas as pd
import sqlite3
import re



# TODO: implement a module which is responsible for
# - Loads the messages and categories datasets
# - Merges the two datasets
# - Cleans the data
# - Stores it in a SQLite database

# - Feel free to use any libraries
# - Yoy should be able to run this file as a script and generate a SQLite database

current_folder = os.getcwd()
sample_file = os.path.join(current_folder, 'disaster_messages.csv')
label_file = os.path.join(current_folder, 'disaster_categories.csv')

email_regex = r'^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'
url_regex = r'(?i)\b((?:https? *:? *(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))'

retweet_regex = r'RT \S+'


def loads_and_merge_data():
    # loads data from csv to dataframes
    df_samples = pd.read_csv(sample_file, delimiter=',')
    # set index as id
    df_samples.set_index('id')
    df_labels = pd.read_csv(label_file, delimiter=',')
    df_labels.set_index('id')
    merged_df = df_samples.merge(df_labels, on='id',left_index=True, right_index=True)
    return merged_df


def clean_data(df):
    # replace email with empty string
    df['message'] = df['message'].apply(lambda txt: re.sub(email_regex, '', txt, re.MULTILINE|re.IGNORECASE))
    # replace urls with empty strings
    df['message'] = df['message'].apply(lambda txt: re.sub(url_regex, '', txt, re.MULTILINE|re.IGNORECASE))
    # replace for samples whose genre is social, replace the RT which is the retweet
    df.loc[(df.genre == 'social'), 'message'] = df.loc[(df.genre == 'social'),'message'].apply(lambda txt: re.sub(retweet_regex, '', txt, re.MULTILINE))
    # repalce @ and # with empty strings
    df['message'] = df['message'].apply(lambda txt: txt.replace('@', '').replace('#', ''), re.MULTILINE)



def store_data(df):
    conn = sqlite3.connect('./db.sqlite')
    df.to_sql('disaster', conn, if_exists='replace', index=True)




df_data = loads_and_merge_data()
clean_data(df_data)
store_data(df_data)



