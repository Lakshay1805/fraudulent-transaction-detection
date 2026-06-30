from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def full_process_pipeline(num_cols , cat_cols):

    num_pipeline = Pipeline([
        ("scaler" , StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("encoder" , OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num" , num_pipeline ,num_cols),
        ("cat" , cat_pipeline , cat_cols)
    ])

    return full_pipeline