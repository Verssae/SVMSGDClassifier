X = np.array(X)
    Y = np.array(Y)
    X_test = np.array(X_test)
    Y_true = np.array(Y_true)
    # normal = Normalizer()
    # scalar = StandardScaler()
    clf  = SVMSGDClassifier(max_epoch=1000,batch_size=32)
    # pipeline = Pipeline(
    #     [ ('transformer',scalar),('estimator', clf)])

    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # # parameters = {'estimator__C': [ 0.1, 1, 10, 100, 1000, 10000]}

    # clf = GridSearchCV(pipeline, param_grid=parameters,
    #                    n_jobs=-1, cv=skf, verbose=True)
    # clf.fit(X, Y)
    Y_pred = clf.fit(X,Y).predict(X_test)
    # cv_results = pd.DataFrame(clf.cv_results_)
    # print(clf.best_params_)
    # print(clf.best_estimator_)
    # print(clf.best_score_)
    # print(cv_results)
    Y_pred = clf.fit(X,Y).predict(X_test)
    print(metrics.f1_score(Y_true,Y_pred,average='macro'))
    # print(clf.score(X_test, Y_true))
    # print(SVMSGDClassifier(max_epoch=1000,batch_size=30).fit(tr_x, tr_x_lbl).score(tt_x, tt_x_lbl))
