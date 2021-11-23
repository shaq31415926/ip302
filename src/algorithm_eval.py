from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt


def algorithm_eval(models, X_train, y_train):
    """Compares a selection of algorithms, evaluates each of them and summarizes the results
    
    @param models - list of models we are interested in evaluating
    @param X_train - the training features
    @param y_train - the target
    
    
    @return a dict of the evaluation metrics and a box plot of the accuracy for every iteration of the cross-validation
    """
    
    
    scoring_results = []
    acc_results = []
    f1_results = []
    names = []

    seed = 7  # set seed to make sure each algorithm is evaluated on the same data

    for name, model in models:
        # evaluate each model using cross-validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        cv_acc_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring='accuracy')
        cv_f1_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring='f1_weighted')
        cv_precision_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring='precision_weighted')
        cv_recall_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring='recall_weighted')

        # for plotting the boxplots
        acc_results.append(cv_acc_results)
        f1_results.append(cv_f1_results)
        names.append(name)

        # summarizing the evaluation metrics for all the models
        scoring_dict = {'Model Name': name,
                        'Accuracy Mean': round(cv_acc_results.mean(), 2),
                        # 'Accuracy STD': round(cv_acc_results.std(), 2),
                        'F1 Mean': round(cv_f1_results.mean(), 2),
                        # 'F1 STD': round(cv_f1_results.std(), 2),
                        'Precision Mean': round(cv_precision_results.mean(), 2),
                        # 'Precision STD': round(cv_precision_results.std(), 2),
                        'Recall Mean': round(cv_recall_results.mean(), 2),
                        # 'Recall STD': round(cv_recall_results.std(), 2)
                        }

        scoring_results.append(scoring_dict)

        print("-" * 50)
        print("{} Accuracy: {} (+/- {})".format(name, round(cv_acc_results.mean(), 5), round(cv_acc_results.std(), 5)))
        print("{} F1-Score: {} (+/- {})".format(name, round(cv_f1_results.mean(), 5), round(cv_f1_results.std(), 5)))
        print("-" * 50)

    # boxplot to summarize the model performance
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Algorithm Accuracy Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(acc_results)
    ax.set_xticklabels(names)
    plt.show()

    return scoring_results