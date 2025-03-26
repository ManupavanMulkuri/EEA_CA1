from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # Create chained labels (y2_3 and y2_3_4) for Design Choice 1
    df = create_combined_labels(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    return model_predict(data, df, name)

def hierarchical_accuracy_from_combined(y_true_df, y_pred):
    total=0
    for i, pred in enumerate(y_pred):
        true = y_true_df.iloc[i]
        try:
            p2, p3, p4 = pred.split('_')
        except ValueError:
            total += 0
            continue

        if p2 != true['y2']:
            score = 0
        elif p3 != true['y3']:
            score = 1/3
        elif p4 != true['y4']:
            score = 2/3
        else:
            score = 1.0
        total += score
    return total / len(y_pred) if len(y_pred) > 0 else 0

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        for label in Config.CHAINED_LABELS:
            temp_df = group_df.copy()
            temp_df['y'] = group_df[label]  # create `y` as target
            data = get_data_object(X, temp_df)
            # label_df = group_df[[label]].rename(columns={label: "y"})
            # data = get_data_object(X, label_df)
            print(f"\n→ Training model for label: {label}")
            model = perform_modelling(data, group_df, name +'-'+label)
            if label == 'y2_3_4':
                y_true_df =  data.get_test_df() # true labels for evaluation
                hierarchical_acc = hierarchical_accuracy_from_combined(y_true_df, model.predictions)
                print(f"Hierarchical Accuracy for group {name}: {hierarchical_acc*100:.2f}%")
                print("\n--- Individual Predictions vs true values for reverification of hierarchical accuracy calculation---")
                for i, pred in enumerate(model.predictions):
                    try:
                        p2, p3, p4 = pred.split('_')
                    except ValueError:
                        p2, p3, p4 = ("INVALID", "INVALID", "INVALID")

                    t2 = y_true_df.iloc[i]["y2"]
                    t3 = y_true_df.iloc[i]["y3"]
                    t4 = y_true_df.iloc[i]["y4"]

                    print(f"[{i+1}] Predicted: ({p2}, {p3}, {p4}) | True: ({t2}, {t3}, {t4})")

