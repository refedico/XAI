### code
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=iris.feature_names, class_names=iris.target_names)

instance_idx = 0
instance = X_test[instance_idx]

explanation = explainer.explain_instance(instance, clf.predict_proba)

print("Spiegazione per l'istanza di test", instance_idx)
print(explanation.as_list())
