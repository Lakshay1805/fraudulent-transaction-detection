
def feature_extraction(df):
    df['destType'] = df['nameDest'].str[0]
    df['origBalanceDiff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['destBalanceDiff'] = df['newbalanceDest'] - df['oldbalanceDest']
    return df

def feature_selection(df):
    df.drop(columns=["nameOrig" , "nameDest" , "isFlaggedFraud"] , inplace=True)
    return df

def full_feature_pipeline(df):
    df = feature_extraction(df)
    df = feature_selection(df)

    return df