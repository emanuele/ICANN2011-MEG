import numpy as np
import scipy.io as io
import preprocessing
import mvpa.suite as ms
from mvpa.clfs.svm import sg

if __name__=='__main__':

    Fs = 200 # 200 Hz
    filter_bank = None
    domain_adaptation = True
    k = 3
    NFFT = 64
    noverlap = 0
    halve = True
    PSD = True
    submission = True
    normalization_single_day = True
    nfolds_CV = 10

    try:
        xd1h_psd
        xd2h_psd
        xd2h_test_psd
    except NameError:
        x_day1_filename = 'megicann_train_v2_day1_'+str(filter_bank)+'.npy'
        print "Loading", x_day1_filename
        x_day1 = np.load(x_day1_filename) # trials X channels X time ; 200Hz.
        y_day1_filename = 'megicann_train_v2_class_day1.npy'
        print "Loading", y_day1_filename
        y_day1 = np.load(y_day1_filename)
        x_day2_filename = 'megicann_train_v2_day2_'+str(filter_bank)+'.npy'
        print "Loading", x_day2_filename
        x_day2 = np.load(x_day2_filename)
        y_day2_filename = 'megicann_train_v2_class_day2.npy'
        print "Loading", y_day2_filename
        y_day2 = np.load(y_day2_filename)
        x_day2_test_filename = 'megicann_test_v2_'+str(filter_bank)+'.npy'
        print "Loading", x_day2_test_filename
        x_day2_test = np.load(x_day2_test_filename)
        
        if halve:
            xd1h = preprocessing.halve_channels(x_day1)
            xd2h = preprocessing.halve_channels(x_day2)
            xd2h_test = preprocessing.halve_channels(x_day2_test)
        else:
            xd1h = x_day1
            xd2h = x_day2
            xd2h_test = x_day2_test
        if PSD:
            xd1h_psd, freq = preprocessing.compute_psd(xd1h, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
            xd2h_psd, freq = preprocessing.compute_psd(xd2h, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
            xd2h_test_psd, freq = preprocessing.compute_psd(xd2h_test, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
        else:
            xd1h_psd = xd1h
            xd2h_psd = xd2h
            xd2h_test_psd = xd2h_test

    # Flatten PSDs or timepoints of all channels in a single vector
    # for each trial.
    samples_day1 = xd1h_psd.reshape(xd1h_psd.shape[0], xd1h_psd.shape[1]*xd1h_psd.shape[2])
    samples_day2 = xd2h_psd.reshape(xd2h_psd.shape[0], xd2h_psd.shape[1]*xd2h_psd.shape[2])
    samples_day2_test = xd2h_test_psd.reshape(xd2h_test_psd.shape[0], xd2h_test_psd.shape[1]*xd2h_test_psd.shape[2])

    # Normalize data of each day individually before building the final dataset:
    if normalization_single_day:
        print "Single Day Normalization."
        samples_day1 -= samples_day1.mean(0)
        samples_day1 /= samples_day1.std(0)
        samples_day1 = np.nan_to_num(samples_day1)
        mean_day2 = np.vstack([samples_day2, samples_day2_test]).mean(0)
        std_day2 = np.vstack([samples_day2, samples_day2_test]).std(0)
        samples_day2 -= mean_day2
        samples_day2 /= std_day2
        samples_day2 = np.nan_to_num(samples_day2)
        samples_day2_test -= mean_day2
        samples_day2_test /= std_day2
        samples_day2_test = np.nan_to_num(samples_day2_test)

    # Build dataset:
    if domain_adaptation:
        # Build dataset with Easy Domain Adaptation:
        samples = np.vstack([np.hstack([samples_day1, samples_day1, np.zeros(samples_day1.shape)]), np.hstack([samples_day2, np.zeros(samples_day2.shape), samples_day2])])
        samples_test = np.hstack([samples_day2_test, np.zeros(samples_day2_test.shape), samples_day2_test])
    else:
        samples = np.vstack([samples_day1, samples_day2])
        samples_test = samples_day2_test
    y = np.concatenate([y_day1, y_day2])

    if not normalization_single_day:
        print "Global Normalization."
        mean_global = np.vstack([samples, samples_test]).mean(0)
        std_global = np.vstack([samples, samples_test]).std(0)
        samples -= mean_global
        samples /= std_global
        samples = np.nan_to_num(samples)
        samples_test -= mean_global
        samples_test /= std_global
        samples_test = np.nan_to_num(samples_test)

    dataset = ms.dataset_wizard(samples=samples, targets=y, chunks=range(samples.shape[0]))
    print "Dataset:", dataset

    C_range = [1.0e1, 1.0e2, 1.0e3]
    sigma2_range = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    C_best = None
    sigma2_best = None
    error_best = 1.0
    for C in C_range:
        for sigma2 in sigma2_range:
            # SVM with Rbf kernel via shogun:
            kernel = ms.RbfKernel(sigma=sigma2)
            clf = sg.SVM(C=C, kernel=kernel)
            # # SVM with Rbf kernel via libsvm (much slower):
            # clf = ms.RbfCSVMC(C=C)
            # clf.kernel_params.gamma=1.0/sigma2
            print "C = %s , sigma2 = %s" % (C, sigma2),
            partitioner = ms.NGroupPartitioner(ngroups=nfolds_CV, attr='chunks')
            cv = ms.CrossValidation(clf, partitioner, enable_ca=['stats'])
            cv_results = cv(dataset)
            cv_error = np.mean(cv_results)
            print "error =", cv_error,
            if cv_error < error_best:
                error_best = cv_error
                C_best = C
                sigma2_best = sigma2
                print "BEST!"
            else:
                print

    print
    print "Model selection ended. Best parameters:"
    print "C = %s , sigma2 = %s" % (C_best, sigma2_best)
    print "Giving CV error =", error_best
    C = C_best
    sigma2 = sigma2_best
    kernel = ms.RbfKernel(sigma=sigma2)
    clf = sg.SVM(C=C, kernel=kernel)

    if submission:
        clf.train(dataset)
        y_test_predicted = np.array(clf.predict(samples_test), dtype=np.int)
        print "Predicted class histogram: ", np.histogram(y_test_predicted, bins=[1,2,3,4,5,6])
        filename = 'class_olivettiDA'
        mdict = {'class_test_day2': y_test_predicted}
        print "Saving predictions to %s.mat/.txt" % filename
        io.savemat(filename+'.mat', mdict)
        np.savetxt(filename+'.txt', y_test_predicted, delimiter='\n', fmt='%d')
    else:
        # Resubstitution error:
        clf.train(dataset)
        y_predicted = clf.predict(dataset.samples)
        resubstitution_error = 1.0 - np.mean(dataset.sa.targets == y_predicted)
        print "resubstitution error =", resubstitution_error
        
        # CV error:
        # partitioner = ms.OddEvenPartitioner(attr='chunks')
        # partitioner = ms.HalfPartitioner(attr='chunks')
        # partitioner = ms.NFoldPartitioner(attr='chunks')
        partitioner = ms.NGroupPartitioner(ngroups=nfolds_CV, attr='chunks')
        cv = ms.CrossValidation(clf, partitioner, enable_ca=['stats'])
        cv_results = cv(dataset)
        cv_error = np.mean(cv_results)
        # print "%s-fold CV error = %s" % (nfolds_CV, cv_error)
        print "CV error =", cv_error
        # print cv.ca.stats.as_string(description=True)
        # print cv_results.samples
        
