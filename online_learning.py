from creme import stream, naive_bayes, neighbors, linear_model, metrics, compose, preprocessing, tree, model_selection, optim
import pandas as pd
import random
import statistics
import numpy as np
from Doctor import Doctor

models = {
    "Decision tree": tree.DecisionTreeClassifier(
        criterion='gini',
        patience=100,
        max_depth=10,
        min_split_gain=0.0,
        min_child_samples=20,
        confidence=1e-5,
        curtail_under=10
    )
    # "Logistic regression": compose.Pipeline(
    #     preprocessing.StandardScaler(),
    #     linear_model.LogisticRegression()
    # ),
    # "Random forest": tree.RandomForestClassifier(
    #     n_trees=10,
    #     seed=42,
    #     # Tree parameters
    #     patience=100,
    #     confidence=1e-5,
    #     criterion='gini'
    # ),
    # "Naive bayes": naive_bayes.GaussianNB(),
    # "KNN": compose.Pipeline(
    #     preprocessing.StandardScaler(),
    #     neighbors.KNeighborsClassifier()
    # ),
    # "ALMA": compose.Pipeline(
    #     preprocessing.StandardScaler(),
    #     linear_model.ALMAClassifier()
    # )
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


def get_points_for_age(age, gender):
    if age < 12775:
        return -1 if gender == 0 else -9
    elif age in range(12775, 14600):
        return 0 if gender == 0 else -4
    elif age in range(14600, 16425):
        return 1 if gender == 0 else 0
    elif age in range(16425, 18250):
        return 2 if gender == 0 else 3
    elif age in range(18250, 20075):
        return 3 if gender == 0 else 6
    elif age in range(20075, 21900):
        return 4 if gender == 0 else 7
    elif age in range(21900, 23725):
        return 5 if gender == 0 else 8
    elif age in range(23725, 25550):
        return 6 if gender == 0 else 8
    elif age >= 25550:
        return 7 if gender == 0 else 8


def get_points_for_cholesterol(cholesterol):
    # If cholesterol is normal
    if cholesterol == 1:
        return 0
    # If cholesterol is above normal
    elif cholesterol == 2:
        return 1
    # If cholesterol is well above normal
    elif cholesterol == 3:
        return 3


def get_points_for_blood_pressure(ap_hi, ap_lo, gender):
    ap_hi_points = 0
    ap_lo_points = 0
    # Male
    if gender == 0:
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
    # Female
    elif gender == 1:
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


def get_feedback(y_pred, points, gender):
    # Male
    if gender == 0:
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
    # Female
    elif gender == 1:
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


def check_relative_risks_and_update_model(model, metric, patient, prediction, true_value, feedback):
    # feedback = 4 or 5
    if feedback >= 4:
        model.fit_one(patient, prediction)
        metric.update(true_value, prediction)
    elif feedback == 3:
        if prediction:
            if get_points_for_blood_pressure(patient["ap_hi"], patient["ap_lo"], patient["gender"]) >= 1:
                model.fit_one(patient, prediction)
                metric.update(true_value, prediction)
            # elif get_points_for_cholesterol(patient["cholesterol"]) >= 0:
            #     model.fit_one(patient, prediction)
            #     metric.update(true_value, prediction)
            else:
                model.fit_one(patient, not prediction)
                metric.update(true_value, not prediction)
        else:
            if get_points_for_blood_pressure(patient["ap_hi"], patient["ap_lo"], patient["gender"]) >= 1:
                model.fit_one(patient, not prediction)
                metric.update(true_value, not prediction)
            # elif get_points_for_cholesterol(patient["cholesterol"]) >= 0:
            #     model.fit_one(patient, not prediction)
            #     metric.update(true_value, not prediction)
            else:
                model.fit_one(patient, prediction)
                metric.update(true_value, prediction)
    elif feedback == 2:
        if prediction:
            if get_points_for_blood_pressure(patient["ap_hi"], patient["ap_lo"], patient["gender"]) >= 2:
                model.fit_one(patient, prediction)
                metric.update(true_value, prediction)
            # elif get_points_for_cholesterol(patient["cholesterol"]) >= 1:
            #     model.fit_one(patient, prediction)
            #     metric.update(true_value, prediction)
            else:
                model.fit_one(patient, not prediction)
                metric.update(true_value, not prediction)
        else:
            if get_points_for_blood_pressure(patient["ap_hi"], patient["ap_lo"], patient["gender"]) >= 2:
                model.fit_one(patient, not prediction)
                metric.update(true_value, not prediction)
            # elif get_points_for_cholesterol(patient["cholesterol"]) >= 1:
            #     model.fit_one(patient, not prediction)
            #     metric.update(true_value, not prediction)
            else:
                model.fit_one(patient, prediction)
                metric.update(true_value, prediction)
    elif feedback == 1:
        if prediction:
            if get_points_for_blood_pressure(patient["ap_hi"], patient["ap_lo"], patient["gender"]) >= 3:
                model.fit_one(patient, prediction)
                metric.update(true_value, prediction)
            # elif get_points_for_cholesterol(patient["cholesterol"]) >= 3:
            #     model.fit_one(patient, prediction)
            #     metric.update(true_value, prediction)
            else:
                model.fit_one(patient, not prediction)
                metric.update(true_value, not prediction)
        else:
            if get_points_for_blood_pressure(patient["ap_hi"], patient["ap_lo"], patient["gender"]) >= 3:
                model.fit_one(patient, not prediction)
                metric.update(true_value, not prediction)
            # elif get_points_for_cholesterol(patient["cholesterol"]) >= 3:
            #     model.fit_one(patient, not prediction)
            #     metric.update(true_value, not prediction)
            else:
                model.fit_one(patient, prediction)
                metric.update(true_value, prediction)
    else:
        model.fit_one(patient, not prediction)
        metric.update(true_value, not prediction)


if __name__ == '__main__':
    doctors = Doctor()
    evaluation_data = {}
    for doctor in doctors.data.itertuples():
        print("Doctor", doctor.id, "Feedback:", doctor.feedback)
        evaluation_data["doctor" + str(doctor.id)] = {"id": doctor.id, "rating": doctor.rating, "reputation": doctor.reputation, "active": doctor.active, "patients": {}, "good_doctor": False}
        # X_y = stream.iter_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned.csv', target='cardio', **params)
        X_y = stream.iter_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned_initial.csv', target='cardio', **params)
        metric = metrics.Accuracy() + metrics.Precision() + metrics.Recall() + metrics.F1()
        models = {
            "Decision tree": tree.DecisionTreeClassifier(
                criterion='gini',
                patience=100,
                max_depth=10,
                min_split_gain=0.0,
                min_child_samples=20,
                confidence=1e-5,
                curtail_under=10
            )
            # "Logistic regression": compose.Pipeline(
            #     preprocessing.StandardScaler(),
            #     linear_model.LogisticRegression()
            # ),
            # "Random forest": tree.RandomForestClassifier(
            #     n_trees=10,
            #     seed=42,
            #     # Tree parameters
            #     patience=100,
            #     confidence=1e-5,
            #     criterion='gini'
            # ),
            # "Naive bayes": naive_bayes.GaussianNB(),
            # "KNN": compose.Pipeline(
            #     preprocessing.StandardScaler(),
            #     neighbors.KNeighborsClassifier()
            # ),
            # "ALMA": compose.Pipeline(
            #     preprocessing.StandardScaler(),
            #     linear_model.ALMAClassifier()
            # )
            # "SVM": svm,
        }
        for model in models:
            print(model)
            # X_y = stream.iter_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned.csv', target='cardio', **params)
            X_y = stream.iter_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned_initial.csv', target='cardio', **params)
            X1_y1 = stream.iter_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned_feedback_patients.csv', target='cardio', **params)
            metric = metrics.Accuracy() + metrics.Precision() + metrics.Recall() + metrics.F1()

            print(model_selection.progressive_val_score(X_y, models[model], metric, print_every=10000))
            # print(model_selection.progressive_val_score(X1_y1, models[model], metric))

            # df = pd.read_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned.csv', sep=",")
            df = pd.read_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned_feedback_patients.csv', sep=",")
            # df = pd.read_csv('./input/cardiovascular-disease-dataset/cardio_train_cleaned_initial.csv', sep=",")

            feedback_column = []
            prediction_column = []
            score_column = []
            feedback = -1
            i = 1
            for x, y in X1_y1:
                # Get prediction
                y_pred = models[model].predict_one(x)
                y = bool(y)

                if doctor.feedback == "distributed":
                    # Calculate score from individual features
                    doctor_score = 0
                    doctor_score += get_points_for_age(x["age"], x["gender"])
                    doctor_score += get_points_for_cholesterol(x["cholesterol"])
                    doctor_score += get_points_for_blood_pressure(x["ap_hi"], x["ap_lo"], x["gender"])
                    doctor_score += get_points_for_bmi(x["bmi"])
                    doctor_score += get_points_for_smoke(x["smoke"])
                    # doctor_score += get_points_for_gluc(x["gluc"])

                    # Get feedback based on prediction and calculated score
                    doctor_feedback = get_feedback(y_pred, doctor_score, x["gender"])
                elif doctor.feedback == "positive":
                    doctor_feedback = np.random.choice(np.arange(0, 6), p=[0, 0, 0, 0, 0.25, 0.75])
                elif doctor.feedback == "negative":
                    doctor_feedback = np.random.choice(np.arange(0, 6), p=[0.75, 0.25, 0, 0, 0, 0])
                elif doctor.feedback == "neutral":
                    doctor_feedback = np.random.choice(np.arange(0, 6), p=[0, 0, 0.35, 0.5, 0.15, 0])
                elif doctor.feedback == "random":
                    doctor_feedback = np.random.choice(np.arange(0, 6), p=[0.16, 0.17, 0.16, 0.17, 0.17, 0.17])
                else:
                    doctor_feedback = -1

                # Calculate score from individual features
                score = 0
                score += get_points_for_age(x["age"], x["gender"])
                score += get_points_for_cholesterol(x["cholesterol"])
                score += get_points_for_blood_pressure(x["ap_hi"], x["ap_lo"], x["gender"])
                score += get_points_for_bmi(x["bmi"])
                score += get_points_for_smoke(x["smoke"])
                # score += get_points_for_gluc(x["gluc"])

                # Get feedback based on prediction and calculated score
                feedback = get_feedback(y_pred, score, x["gender"])
                score_column.append(score)
                prediction_column.append(y_pred)
                feedback_column.append(feedback)

                if i < 1000:
                    evaluation_data["doctor" + str(doctor.id)]["patients"][i] = {"patient_data": x,
                                                                                 "prediction": y_pred, "true_value": y,
                                                                                 "doctor_feedback": doctor_feedback,
                                                                                 "our_feedback": feedback}
                elif i == 1000:
                    evaluation_data["doctor" + str(doctor.id)]["patients"][i] = {"patient_data": x,
                                                                                 "prediction": y_pred, "true_value": y,
                                                                                 "doctor_feedback": doctor_feedback,
                                                                                 "our_feedback": feedback}
                    evaluation_data["doctor" + str(doctor.id)]["good_doctor"] = doctors.evaluate_doctor(evaluation_data["doctor" + str(doctor.id)])
                    if evaluation_data["doctor" + str(doctor.id)]["good_doctor"]:
                        for patient_id, patient in evaluation_data["doctor" + str(doctor.id)]["patients"].items():
                            check_relative_risks_and_update_model(models[model], metric, patient["patient_data"], patient["prediction"], patient["true_value"], patient["doctor_feedback"])
                    else:
                        for patient_id, patient in evaluation_data["doctor" + str(doctor.id)]["patients"].items():
                            check_relative_risks_and_update_model(models[model], metric, patient["patient_data"], patient["prediction"], patient["true_value"], patient["our_feedback"])
                elif evaluation_data["doctor" + str(doctor.id)]["good_doctor"]:
                    check_relative_risks_and_update_model(models[model], metric, x, y_pred, y, doctor_feedback)
                else:
                    check_relative_risks_and_update_model(models[model], metric, x, y_pred, y, feedback)

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

        print()
