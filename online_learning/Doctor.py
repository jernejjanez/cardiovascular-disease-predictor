import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


class Doctor:
    def __init__(self):
        self.data = None

        self.read_data()

    def read_data(self, file_name='./input/cardiovascular-disease-dataset/doctors.csv'):
        """Read data from file and save it to pandas data frame"""

        # Data Cleaning and EDA
        data_raw = pd.read_csv(file_name, sep=",")

        self.data = data_raw
        # self.labels = df.columns

    def evaluate_doctor(self, doctor):
        evaluation_points = 0
        doctor_feedback = []
        our_feedback = []
        same_feedback = 0
        similar_feedback = 0
        for patient_id, patient_info in doctor["patients"].items():
            doctor_feedback.append(patient_info["doctor_feedback"])
            our_feedback.append(patient_info["our_feedback"])

            if patient_info["doctor_feedback"] == patient_info["our_feedback"]:
                same_feedback += 1

            if patient_info["doctor_feedback"] in range(max(patient_info["our_feedback"] - 1, 0), min(patient_info["our_feedback"] + 2, 6)):
                similar_feedback += 1

        doctor_feedback_occurrences = Counter(doctor_feedback)
        doctor_feedback_occurrences_percentage = {}
        for i in range(0, 6):
            if i not in doctor_feedback_occurrences:
                doctor_feedback_occurrences[i] = 0
            doctor_feedback_occurrences_percentage[i] = self.get_percentage(doctor_feedback_occurrences[i], len(doctor_feedback))

        self.visualize_feedback(doctor_feedback_occurrences)

        same_feedback_percentage = self.get_percentage(same_feedback, len(doctor_feedback))
        similar_feedback_percentage = self.get_percentage(similar_feedback, len(doctor_feedback))

        evaluation_points += self.get_points_for_feedback_occurence(doctor_feedback_occurrences_percentage)
        evaluation_points += self.get_points_for_same_feedback(same_feedback_percentage)
        evaluation_points += self.get_points_for_similar_feedback(similar_feedback_percentage)

        print("Rating:", doctor["rating"])
        print("Reputation:", doctor["reputation"])
        print("Activeness:", doctor["active"])
        evaluation_points += doctor["rating"]
        evaluation_points += doctor["reputation"]
        evaluation_points += doctor["active"]

        return self.is_good_doctor(evaluation_points)

    def is_good_doctor(self, eval_points):
        print()
        if eval_points > 11:
            print("Is good doctor? Yes")
            print("Updating model using doctors feedback")
            return True
        print("Is good doctor? No")
        print("Updating model using our feedback")
        return False

    def get_points_for_feedback_occurence(self, feedback_occurrences_percentage):
        for feedback, percentage in feedback_occurrences_percentage.items():
            if percentage > 90:
                return -4

        return 0

    def get_points_for_same_feedback(self, similarity_percentage):
        if similarity_percentage < 25:
            return -2
        elif 25 <= similarity_percentage < 50:
            return 0
        elif 50 <= similarity_percentage < 75:
            return 2
        elif similarity_percentage >= 75:
            return 4

    def get_points_for_similar_feedback(self, similarity_percentage):
        if similarity_percentage < 25:
            return -6
        elif 25 <= similarity_percentage < 50:
            return -4
        elif 50 <= similarity_percentage < 75:
            return -2
        elif similarity_percentage >= 75:
            return 0

    def visualize_feedback(self, feedback_occurrences):
        plt.bar(feedback_occurrences.keys(), feedback_occurrences.values())
        plt.xlabel("Feedback")
        plt.ylabel("Occurrence")
        plt.show()

    def get_percentage(self, current, total):
        return current / total * 100
