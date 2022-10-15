import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn_lvq import GlvqModel
from sklearn.ensemble import RandomForestClassifier


DATASET_NAME = 'full_dataset.csv'
ATTRIBUTES_NAMES = [
    'gender', 
    'C_api', 
    'C_man', 
    'E_NEds', 
    'E_Bpag', 
    'firstDay', 
    'lastDay', 
    'NEds', 
    'NDays', 
    'NActDays', 
    'NPages', 
    'NPcreated', 
    'pagesWomen', 
    'wikiprojWomen', 
    'ns_user', 
    'ns_wikipedia', 
    'ns_talk', 
    'ns_userTalk', 
    'ns_content', 
    'weightIJ', 
    'NIJ',
]


def load_and_preprocess_dataset():
    # load dataset
    dataframe = pandas.read_csv(DATASET_NAME, names=ATTRIBUTES_NAMES)
    dataframe = dataframe.drop(0)
    dataframe = dataframe.apply(
        preprocessing.LabelEncoder().fit_transform)

    array = dataframe.values
    attributes_count = len(ATTRIBUTES_NAMES)
    x = array[:,1:attributes_count]
    y = array[:,0]

    return x, y


def create_models():
    # Make experiments with K-NN, LVQ, Decision Tree, SVM,
    # Random Forest and an heterogen "committee".
    # prepare models

    models_dict = {
        'K-NN': KNeighborsClassifier(),
        'LVQ': GlvqModel(),
        'DecisionTree': DecisionTreeClassifier(),
        'SVM': SVC(),
        'RandomForest': RandomForestClassifier(),
    }

    return [
        (model_name, model)
        for model_name, model in models_dict.items()
    ]


def test_knn(x, y):
    # Test model - predit:
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y)

    # the result must be 1
    # Test one line for testing purposes
    prediction = neigh.predict([[ 
            1, 3, 2, 0, 20130107110046, 20170915092339, 590, 1713, 261, 501, 12, 0, 0, 8, 1, 0, 9, 540, 1.20440316450483, 647
    ]])
    print(prediction)


def cross_validate_models(models, x, y):
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(
            model, x, y, cv=kfold, scoring=scoring
        )
        results.append(cv_results)
        names.append(name)
        print(
            "{}: {} ({})".format(
                name, cv_results.mean(), cv_results.std()
            )
        )

    return results, names


def plot_box_plot(models_results, models_names):
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(models_results)
    ax.set_xticklabels(models_names)
    plt.show()


def main():
    x, y = load_and_preprocess_dataset()

    # Used for checking if the dataset is prepared.
    # test_knn(x, y)

    models_results, models_names = cross_validate_models(
        create_models(), x, y
    )
    plot_box_plot(models_results, models_names)


if __name__ == '__main__':
    main()
