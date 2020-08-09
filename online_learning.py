from creme import stream, naive_bayes, neighbors, linear_model, metrics, compose, preprocessing, tree, model_selection, optim
import pandas as pd
import random
import statistics

models = {
    "Logistic regression": compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression()
    ),
    "Decision tree": tree.DecisionTreeClassifier(
        criterion='gini',
        patience=100,
        max_depth=10,
        min_split_gain=0.0,
        min_child_samples=20,
        confidence=1e-5,
        curtail_under=10
    ),
    "Random forest": tree.RandomForestClassifier(
        n_trees=10,
        seed=42,
        # Tree parameters
        patience=100,
        confidence=1e-5,
        criterion='gini'
    ),
    "Naive bayes": naive_bayes.GaussianNB(),
    "KNN": compose.Pipeline(
        preprocessing.StandardScaler(),
        neighbors.KNeighborsClassifier()
    ),
    "ALMA": compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.ALMAClassifier()
    )
    # "SVM": svm,

}

params = {
    'converters': {'age': int,
                   'gender': int,
                   'height': int,
                   'weight': float,
                   'ap_hi': int,
                   'ap_lo': int,
                   'cholesterol': int,
                   'gluc': int,
                   'smoke': int,
                   'alco': int,
                   'active': int,
                   'cardio': int,
                   'bmi': float}
}


def get_points_for_age_male(age):
    if age < 12775:
        return -1
    elif age in range(12775, 14600):
        return 0
    elif age in range(14600, 16425):
        return 1
    elif age in range(16425, 18250):
        return 2
    elif age in range(18250, 20075):
        return 3
    elif age in range(20075, 21900):
        return 4
    elif age in range(21900, 23725):
        return 5
    elif age in range(23725, 25550):
        return 6
    elif age >= 25550:
        return 7


def get_points_for_age_female(age):
    if age < 12775:
        return -9
    elif age in range(12775, 14600):
        return -4
    elif age in range(14600, 16425):
        return 0
    elif age in range(16425, 18250):
        return 3
    elif age in range(18250, 20075):
        return 6
    elif age in range(20075, 21900):
        return 7
    elif age in range(21900, 23725):
        return 8
    elif age in range(23725, 25550):
        return 8
    elif age >= 25550:
        return 8


def get_points_for_cholesterol_male(cholesterol):
    # If cholesterol is normal
    if cholesterol == 1:
        return 0
    # If cholesterol is above normal
    elif cholesterol == 2:
        return 1
    # If cholesterol is well above normal
    elif cholesterol == 3:
        return 3


def get_points_for_cholesterol_female(cholesterol):
    # If cholesterol is normal
    if cholesterol == 1:
        return 0
    # If cholesterol is above normal
    elif cholesterol == 2:
        return 1
    # If cholesterol is well above normal
    elif cholesterol == 3:
        return 3


def get_points_for_blood_pressure_male(ap_hi, ap_lo):
    ap_hi_points = 0
    if ap_hi < 120:
        ap_hi_points = 0
    elif ap_hi in range(120, 130):
        ap_hi_points = 0
    elif ap_hi in range(130, 140):
        ap_hi_points = 1
    elif ap_hi in range(140, 160):
        ap_hi_points = 2
    elif ap_hi >= 160:
        ap_hi_points = 3

    ap_lo_points = 0
    if ap_lo < 80:
        ap_lo_points = 0
    elif ap_lo in range(80, 85):
        ap_lo_points = 0
    elif ap_lo in range(85, 90):
        ap_lo_points = 1
    elif ap_lo in range(90, 100):
        ap_lo_points = 2
    elif ap_lo >= 100:
        ap_lo_points = 3

    return max(ap_hi_points, ap_lo_points)


def get_points_for_blood_pressure_female(ap_hi, ap_lo):
    ap_hi_points = 0
    if ap_hi < 120:
        ap_hi_points = -1
    elif ap_hi in range(120, 130):
        ap_hi_points = 0
    elif ap_hi in range(130, 140):
        ap_hi_points = 1
    elif ap_hi in range(140, 160):
        ap_hi_points = 2
    elif ap_hi >= 160:
        ap_hi_points = 3

    ap_lo_points = 0
    if ap_lo < 80:
        ap_lo_points = -1
    elif ap_lo in range(80, 85):
        ap_lo_points = 0
    elif ap_lo in range(85, 90):
        ap_lo_points = 1
    elif ap_lo in range(90, 100):
        ap_lo_points = 2
    elif ap_lo >= 100:
        ap_lo_points = 3

    return max(ap_hi_points, ap_lo_points)


def get_points_for_smoke(smoke):
    if smoke:
        return 2
    else:
        return 0


def get_points_for_gluc(gluc):
    # If glucose is normal
    if gluc == 1:
        return 0
    # If glucose is above normal
    elif gluc == 2:
        return 1
    # If glucose is well above normal
    elif gluc == 3:
        return 3


def get_points_for_bmi(bmi):
    # a BMI of 18.5 to 24.9 is considered a normal, or healthy, weight
    if bmi < 25:
        return 0
    # A BMI that ranges from 25 to 29.9 is considered overweight
    elif 25 <= bmi < 30:
        return 1
    # BMI of 30 or higher falls into the obese category, according to the CDC
    # Class 1 obesity is a BMI of 30 to 34.9
    elif 30 <= bmi < 35:
        return 2
    # Class 2 is 35 to 39.9
    elif 35 <= bmi < 40:
        return 3
    # Class 3 is 40 and above
    elif bmi >= 40:
        return 4


def get_feedback_male(y_pred, points):
    if y_pred:
        if points < 0:
            return 0
        elif points in range(0, 2):
            return 1
        elif points in range(2, 4):
            return 2
        elif points in range(4, 7):
            return 3
        elif points in range(7, 9):
            return 4
        elif points >= 9:
            return 5
    else:
        if points < 0:
            return 5
        elif points in range(0, 2):
            return 4
        elif points in range(2, 4):
            return 3
        elif points in range(4, 7):
            return 2
        elif points in range(7, 9):
            return 1
        elif points >= 9:
            return 0


def get_feedback_female(y_pred, points):
    if y_pred:
        if points < 0:
            return 0
        elif points in range(0, 3):
            return 1
        elif points in range(3, 5):
            return 2
        elif points in range(5, 8):
            return 3
        elif points in range(8, 11):
            return 4
        elif points >= 11:
            return 5
    else:
        if points < 0:
            return 5
        elif points in range(0, 3):
            return 4
        elif points in range(3, 5):
            return 3
        elif points in range(5, 8):
            return 2
        elif points in range(8, 11):
            return 1
        elif points >= 11:
            return 0


if __name__ == '__main__':
    X_y = stream.iter_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned.csv', target='cardio', **params)
    # metric = metrics.Accuracy() + metrics.Precision() + metrics.Recall() + metrics.F1() + metrics.LogLoss() + metrics.ROCAUC()
    metric = metrics.Accuracy() + metrics.Precision() + metrics.Recall() + metrics.F1()
    for model in models:
        print(model)
        # X_y = stream.iter_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned.csv', target='cardio', **params)
        X_y = stream.iter_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned_initial.csv', target='cardio', **params)
        X1_y1 = stream.iter_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned_feedback_patients.csv', target='cardio', **params)
        # metric = metrics.Accuracy() + metrics.Precision() + metrics.Recall() + metrics.F1() + metrics.LogLoss() + metrics.ROCAUC()
        metric = metrics.Accuracy() + metrics.Precision() + metrics.Recall() + metrics.F1()

        # for x, y in X_y:
        #     y_pred_proba = models[model].predict_proba_one(x)
        #     if y_pred_proba[True] >= 0.1:
        #         y_pred = 1
        #     else:
        #         y_pred = 0
        #
        #     models[model].fit_one(x, y)
        #     metric.update(y, y_pred)
        #
        # print(metric)

        print(model_selection.progressive_val_score(X_y, models[model], metric, print_every=10000))
        # print(model_selection.progressive_val_score(X1_y1, models[model], metric))

        # df = pd.read_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned.csv', sep=",")
        df = pd.read_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned_feedback_patients.csv', sep=",")
        # df = pd.read_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned_initial.csv', sep=",")

        feedback_column = []
        prediction_column = []
        points_column_male = []
        points_column_female = []
        feedback = -1
        i = 1
        for x, y in X1_y1:
            # y_pred_proba = models[model].predict_proba_one(x)
            # if y_pred_proba[True] >= 0.2:
            #     y_pred = 1
            # else:
            #     y_pred = 0
            y_pred = models[model].predict_one(x)
            y = bool(y)
            points = 0
            # Male
            if x["gender"] == 0:
                points += get_points_for_age_male(x["age"])
                points += get_points_for_cholesterol_male(x["cholesterol"])
                points += get_points_for_blood_pressure_male(x["ap_hi"], x["ap_lo"])
                points += get_points_for_bmi(x["bmi"])
                points += get_points_for_smoke(x["smoke"])
                # points += get_points_for_gluc(x["gluc"])

                feedback = get_feedback_male(y_pred, points)
                points_column_male.append(points)

            # Female
            elif x["gender"] == 1:
                points += get_points_for_age_female(x["age"])
                points += get_points_for_cholesterol_female(x["cholesterol"])
                points += get_points_for_blood_pressure_female(x["ap_hi"], x["ap_lo"])
                points += get_points_for_bmi(x["bmi"])
                points += get_points_for_smoke(x["smoke"])
                # points += get_points_for_gluc(x["gluc"])

                feedback = get_feedback_female(y_pred, points)
                points_column_female.append(points)

            prediction_column.append(y_pred)
            feedback_column.append(feedback)

            # feedback = 5
            if feedback >= 4:
                models[model].fit_one(x, y_pred)
                metric.update(y, y_pred)
            elif feedback == 3:
                if x["gender"] == 0:
                    if y_pred:
                        if get_points_for_blood_pressure_male(x["ap_hi"], x["ap_lo"]) >= 1:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
                        # elif get_points_for_cholesterol_male(x["cholesterol"]) >= 0:
                        #     models[model].fit_one(x, y_pred)
                        #     metric.update(y, y_pred)
                        else:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                    else:
                        if get_points_for_blood_pressure_male(x["ap_hi"], x["ap_lo"]) >= 1:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                        # elif get_points_for_cholesterol_male(x["cholesterol"]) >= 0:
                        #     models[model].fit_one(x, not y_pred)
                        #     metric.update(y, not y_pred)
                        else:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
                else:
                    if y_pred:
                        if get_points_for_blood_pressure_female(x["ap_hi"], x["ap_lo"]) >= 1:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
                        # elif get_points_for_cholesterol_female(x["cholesterol"]) >= 0:
                        #     models[model].fit_one(x, y_pred)
                        #     metric.update(y, y_pred)
                        else:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                    else:
                        if get_points_for_blood_pressure_female(x["ap_hi"], x["ap_lo"]) >= 1:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                        # elif get_points_for_cholesterol_female(x["cholesterol"]) >= 0:
                        #     models[model].fit_one(x, not y_pred)
                        #     metric.update(y, not y_pred)
                        else:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
            elif feedback == 2:
                if x["gender"] == 0:
                    if y_pred:
                        if get_points_for_blood_pressure_male(x["ap_hi"], x["ap_lo"]) >= 2:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
                        # elif get_points_for_cholesterol_male(x["cholesterol"]) >= 1:
                        #     models[model].fit_one(x, y_pred)
                        #     metric.update(y, y_pred)
                        else:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                    else:
                        if get_points_for_blood_pressure_male(x["ap_hi"], x["ap_lo"]) >= 2:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                        # elif get_points_for_cholesterol_male(x["cholesterol"]) >= 1:
                        #     models[model].fit_one(x, not y_pred)
                        #     metric.update(y, not y_pred)
                        else:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
                else:
                    if y_pred:
                        if get_points_for_blood_pressure_female(x["ap_hi"], x["ap_lo"]) >= 2:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
                        # elif get_points_for_cholesterol_female(x["cholesterol"]) >= 1:
                        #     models[model].fit_one(x, y_pred)
                        #     metric.update(y, y_pred)
                        else:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                    else:
                        if get_points_for_blood_pressure_female(x["ap_hi"], x["ap_lo"]) >= 2:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                        # elif get_points_for_cholesterol_female(x["cholesterol"]) >= 1:
                        #     models[model].fit_one(x, not y_pred)
                        #     metric.update(y, not y_pred)
                        else:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
            elif feedback == 1:
                if x["gender"] == 0:
                    if y_pred:
                        if get_points_for_blood_pressure_male(x["ap_hi"], x["ap_lo"]) >= 3:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
                        # elif get_points_for_cholesterol_male(x["cholesterol"]) >= 3:
                        #     models[model].fit_one(x, y_pred)
                        #     metric.update(y, y_pred)
                        else:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                    else:
                        if get_points_for_blood_pressure_male(x["ap_hi"], x["ap_lo"]) >= 3:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                        # elif get_points_for_cholesterol_male(x["cholesterol"]) >= 3:
                        #     models[model].fit_one(x, not y_pred)
                        #     metric.update(y, not y_pred)
                        else:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
                else:
                    if y_pred:
                        if get_points_for_blood_pressure_female(x["ap_hi"], x["ap_lo"]) >= 3:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
                        # elif get_points_for_cholesterol_female(x["cholesterol"]) >= 3:
                        #     models[model].fit_one(x, y_pred)
                        #     metric.update(y, y_pred)
                        else:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                    else:
                        if get_points_for_blood_pressure_female(x["ap_hi"], x["ap_lo"]) >= 3:
                            models[model].fit_one(x, not y_pred)
                            metric.update(y, not y_pred)
                        # elif get_points_for_cholesterol_female(x["cholesterol"]) >= 3:
                        #     models[model].fit_one(x, not y_pred)
                        #     metric.update(y, not y_pred)
                        else:
                            models[model].fit_one(x, y_pred)
                            metric.update(y, y_pred)
            else:
                models[model].fit_one(x, not y_pred)
                metric.update(y, not y_pred)
            # elif feedback == 4 and x["age"] > 20075:
            #     models[model].fit_one(x, y_pred)
            #     metric.update(y, y_pred)
            # elif feedback == 3 and x["age"] > 21900:
            #     models[model].fit_one(x, y_pred)
            #     metric.update(y, y_pred)
            # elif feedback == 2 and x["age"] > 23725:
            #     models[model].fit_one(x, y_pred)
            #     metric.update(y, y_pred)
            # elif feedback == 1 and x["age"] > 25550:
            #     models[model].fit_one(x, y_pred)
            #     metric.update(y, y_pred)
            # else:
            #     models[model].fit_one(x, not y_pred)
            #     metric.update(y, not y_pred)

            if i % 10000 == 0:
                print(metric)

            i += 1

        df["y_pred"] = prediction_column
        df["feedback"] = feedback_column

        print(metric)
        # print("Male points mean:", statistics.mean(points_column_male))
        # print("Female points mean:", statistics.mean(points_column_female))
        # print("Male points median:", statistics.median(points_column_male))
        # print("Female points median:", statistics.median(points_column_female))
        # print("Feedback mean:", statistics.mean(feedback_column))
        # print("Feedback median:", statistics.median(feedback_column))
