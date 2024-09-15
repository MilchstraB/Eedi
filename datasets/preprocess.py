import pandas as pd


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    # Melt the answer columns
    answers_df = pd.melt(
        id_vars=keep_cols,
        frame=df[keep_cols + answer_cols],
        var_name='Answer', value_name='Value'
    ).sort_values(["QuestionId", "Answer"]).reset_index(drop=True)
    
    # If NOT test set
    if misconception_cols[0] in df.columns:
        
        # Melt the misconception columns
        misconceptions_df = pd.melt(
            id_vars=keep_cols,
            frame=df[keep_cols + misconception_cols],
            var_name='Misconception', value_name='MisconceptionId'
        ).sort_values(["QuestionId", "Misconception"]).reset_index(drop=True)

        answers_df[['Misconception', 'MisconceptionId']] = misconceptions_df[['Misconception', 'MisconceptionId']]
    
    return answers_df


if __name__ == "__main__":
    keep_cols = ["QuestionId", "ConstructName", "SubjectName", "CorrectAnswer", "QuestionText"]
    answer_cols = ["AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText"]
    misconception_cols = ["MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId"]

    train_data = pd.read_csv("data/train.csv")
    train_data = wide_to_long(train_data)
    train_data.to_csv("data/train_after_process.csv", index=False)
