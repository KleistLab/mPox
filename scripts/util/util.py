import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import optimize


def combine_results(repo, samples = 1000):
    frames_D = []
    frames_I = []
    frames_lambda = []
    frames_R = []
    frames_R_lambda = []
    frames_M = []
    frames_mean_degree_I = []
    frames_I_degrees = []
    frames_I_degrees = []



    for index_start in range(samples):
        index_start = index_start * 100
        index_end = index_start + 100
        try:
            file = repo+"/has_result_D_cum_"+str(index_start)+"_"+str(index_end)+".csv"
            data_D_cum = pd.read_csv(file)
            file = repo+"/has_result_I_cum_"+str(index_start)+"_"+str(index_end)+".csv"
            data_I_cum = pd.read_csv(file)
            file = repo+"/has_result_lambda_"+str(index_start)+"_"+str(index_end)+".csv"
            data_lambda = pd.read_csv(file)
            file = repo+"/has_result_R_"+str(index_start)+"_"+str(index_end)+".csv"
            data_R = pd.read_csv(file)
            file = repo+"/has_result_R_lambda_"+str(index_start)+"_"+str(index_end)+".csv"
            data_R_lambda = pd.read_csv(file)
            file = repo+"/has_result_M_"+str(index_start)+"_"+str(index_end)+".csv"
            data_M = pd.read_csv(file)
            file = repo+"/has_result_mean_degree_I_"+str(index_start)+"_"+str(index_end)+".csv"
            data_mean_degree_I = pd.read_csv(file)
            file = repo+"/has_result_I_degree_list_"+str(index_start)+"_"+str(index_end)+".csv"
            data_I_degrees = pd.read_csv(file)
            frames_D.append(data_D_cum)
            frames_I.append(data_I_cum)
            frames_lambda.append(data_lambda)
            frames_R.append(data_R)
            frames_R_lambda.append(data_R_lambda)
            frames_M.append(data_M)
            frames_mean_degree_I.append(data_mean_degree_I)
            frames_I_degrees.append(data_I_degrees)
        except:
            print(index_start)
        

    data_D_all = pd.concat(frames_D)
    data_I_all = pd.concat(frames_I)
    data_lambda_all = pd.concat(frames_lambda)
    data_R = pd.concat(frames_R)
    data_R_lambda = pd.concat(frames_R_lambda)
    data_M = pd.concat(frames_M)
    data_mean_degree_I = pd.concat(frames_mean_degree_I)
    data_I_degrees = pd.concat(frames_I_degrees)

    del frames_D
    del frames_I
    del frames_lambda
    del frames_R
    del frames_R_lambda
    del frames_M
    del frames_mean_degree_I
    del frames_I_degrees

    data_D_all.to_csv(repo+"/data_D_all.csv", index=False)
    data_I_all.to_csv(repo+"/data_I_all.csv", index=False)
    data_lambda_all.to_csv(repo+"/data_lambda_all.csv", index=False)
    data_R.to_csv(repo+"/data_R.csv", index=False)
    data_M.to_csv(repo+"/data_M.csv", index=False)
    data_mean_degree_I.to_csv(repo+"/data_mean_degree_I.csv", index=False)
    data_I_degrees.to_csv(repo+"/data_I_degrees.csv", index=False)
    data_R_lambda.to_csv(repo+"/data_R_lambda.csv", index=False)



def apply_ll(repo, multiple, LL_berlin):
    data_D_all = pd.read_csv(repo+"/data_D_all.csv")
    data_I_all = pd.read_csv(repo+"/data_I_all.csv")
    data_lambda_all = pd.read_csv(repo+"/data_lambda_all.csv")
    data_R = pd.read_csv(repo+"/data_R.csv")
    data_M = pd.read_csv(repo+"/data_M.csv")
    data_mean_degree_I = pd.read_csv(repo+"/data_mean_degree_I.csv")
    data_I_degrees = pd.read_csv(repo+"/data_I_degrees.csv")
    data_R_lambda = pd.read_csv(repo+"/data_R_lambda.csv")


    LL_range_D = data_D_all[(data_D_all["LL"] > multiple* LL_berlin)&(data_D_all["LL"] <LL_berlin)]
    LL_range_I = data_I_all[(data_I_all["LL"] > multiple* LL_berlin)&(data_I_all["LL"] < LL_berlin)]
    LL_range_lambda = data_lambda_all[(data_lambda_all["LL"] > multiple* LL_berlin)&(data_lambda_all["LL"] < LL_berlin)]
    LL_range_R = data_R[(data_R["LL"] > multiple* LL_berlin)&(data_R["LL"] < LL_berlin)]
    LL_range_R_lambda = data_R_lambda[(data_R_lambda["LL"] > multiple* LL_berlin)&(data_R_lambda["LL"] < LL_berlin)]
    LL_range_M = data_M[(data_M["LL"] > multiple* LL_berlin)&(data_M["LL"] < LL_berlin)]
    LL_range_mean_degree_I = data_mean_degree_I[(data_mean_degree_I["LL"] > multiple* LL_berlin)&(data_mean_degree_I["LL"] < LL_berlin)]
    LL_range_I_degrees = data_I_degrees[(data_I_degrees["LL"] > multiple* LL_berlin)&(data_I_degrees["LL"] < LL_berlin)]

    print(len(LL_range_D))

    return(LL_range_D, LL_range_I, LL_range_lambda, LL_range_R, LL_range_R_lambda, LL_range_M, LL_range_mean_degree_I, LL_range_I_degrees)


def apply_ll_D(repo, multiple, LL_berlin):
    data_D_all = pd.read_csv(repo+"/data_D_all.csv")
    
    LL_range_D = data_D_all[(data_D_all["LL"] > multiple* LL_berlin)&(data_D_all["LL"] <LL_berlin)]
    
    print(len(LL_range_D))

    return(LL_range_D)


# timelines

def plot_timelines(LL_range_D, LL_range_R, LL_range_lambda, LL_range_M, LL_range_mean_degree_I, LL_range_R_lambda, N, mean_infectious, berlin):
    counter = 25
    xlim = counter

    cols = 3
    rows = 2

    fig, ax = plt.subplots(rows,cols, figsize = [cols * 4, rows * 4])#[len(cols)*6.4, 4.8])

    t = np.linspace(0,25,26)
    start_index = 5
    end_index = 31


    # diagnosed
    ax[0,0].plot(np.mean(LL_range_D, axis = 0)[5:31], color = "darkblue", linestyle = "dashed")
    ax[0,0].fill_between(t,LL_range_D.quantile(0.025)[5:31], LL_range_D.quantile(0.975)[5:31], color = "lightblue", alpha = 0.9)

    # recovered
    ax[1,1].plot(np.mean(LL_range_R, axis = 0)[5:31]/N*100, color = "darkblue", linestyle = "dashed")
    ax[1,1].fill_between(t,LL_range_R.quantile(0.025)[5:31]/N*100, LL_range_R.quantile(0.975)[5:31]/N*100, color = "lightblue", alpha = 0.9)

    # contact reduction
    ax[1,0].plot(np.mean(LL_range_lambda, axis = 0)[5:31]/LL_range_lambda.iloc[0][start_index]*100, color = "darkblue", linestyle = "dashed")
    ax[1,0].fill_between(t,LL_range_lambda.quantile(0.025)[5:31]/LL_range_lambda.iloc[0][start_index]*100, LL_range_lambda.quantile(0.975)[5:31]/LL_range_lambda.iloc[0][start_index]*100, color = "lightblue", alpha = 0.9)

    # contact with immune agents
    ax[0,1].plot(np.mean(LL_range_R_lambda.iloc[:,5:31].multiply(1/LL_range_lambda.iloc[:,5:31], axis = "index"))/2*100, color = "darkblue", linestyle = "dashed")
    ax[0,1].fill_between(t, LL_range_R_lambda.iloc[:,5:31].multiply(1/LL_range_lambda.iloc[:,5:31], axis = "index").quantile(0.025)/2*100, LL_range_R_lambda.iloc[:,5:31].multiply(1/LL_range_lambda.iloc[:,5:31], axis = "index").quantile(0.975)/2*100, color = "lightblue", alpha = 0.9)

    # measure activations
    ax[0,2].plot(np.mean(np.cumsum(LL_range_M.iloc[:, 5:31], axis = 1)/N*100), color = "darkblue", linestyle = "dashed")
    ax[0,2].fill_between(t,np.cumsum(LL_range_M.iloc[:, 5:31], axis = 1).quantile(0.025)/N*100, np.cumsum(LL_range_M.iloc[:, 5:31], axis = 1).quantile(0.975)/N*100, color = "lightblue", alpha = 0.9)

    # R0 correlate
    ax[1,2].plot(np.mean(LL_range_mean_degree_I.iloc[:,5:31].multiply(LL_range_mean_degree_I["inf_prob"], axis="index")*mean_infectious), color = "darkblue", linestyle = "dashed")
    ax[1,2].fill_between(t,LL_range_mean_degree_I.iloc[:,5:31].multiply(LL_range_mean_degree_I["inf_prob"], axis="index").quantile(0.025)*mean_infectious, LL_range_mean_degree_I.iloc[:,5:31].multiply(LL_range_mean_degree_I["inf_prob"], axis="index").quantile(0.975)*mean_infectious, color = "lightblue", alpha = 0.9)

    ax[0,0].plot(berlin[:xlim+1], color = "forestgreen", linewidth = 2)
    ax[0,0].plot(np.mean(LL_range_D, axis = 0)[start_index:end_index], color = "darkblue", linestyle = "dashed")

    #ax[1,2].plot(mean/len(LL_range_D), color = "darkblue", linestyle = "dashed")
    ax[1,2].axhline(1, color = "black", linestyle = "dashed")

    # cosmetics
    ax[0,0].set_xticks(np.linspace(0,25, 26), ["KW"+str(i) for i in range(18,44)], rotation=90)
    ax[0,1].set_xticks(np.linspace(0,25, 26), ["KW"+str(i) for i in range(18,44)], rotation=90)
    ax[0,2].set_xticks(np.linspace(0,25, 26), ["KW"+str(i) for i in range(18,44)], rotation=90)
    ax[1,0].set_xticks(np.linspace(0,25, 26), ["KW"+str(i) for i in range(18,44)], rotation=90)
    ax[1,1].set_xticks(np.linspace(0,25, 26), ["KW"+str(i) for i in range(18,44)], rotation=90)
    ax[1,2].set_xticks(np.linspace(0,25, 26), ["KW"+str(i) for i in range(18,44)], rotation=90)

    ax[0,0].spines['top'].set_visible(False)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,1].spines['top'].set_visible(False)
    ax[0,1].spines['right'].set_visible(False)
    ax[0,2].spines['top'].set_visible(False)
    ax[0,2].spines['right'].set_visible(False)
    ax[1,0].spines['top'].set_visible(False)
    ax[1,0].spines['right'].set_visible(False)
    ax[1,1].spines['top'].set_visible(False)
    ax[1,1].spines['right'].set_visible(False)


    ax[0,0].title.set_text("Diagnosed cases per week")
    ax[1,0].title.set_text("Realized contacts\n as share of expected contacts (%)")
    ax[1,1].title.set_text("Recovered share of population (%)")
    ax[0,1].title.set_text("Contacts involving at least\none immune agent (%)")
    ax[0,2].title.set_text("People with reduced bahaviour")
    ax[1,2].title.set_text(r"$R_0$")

    ax[1,2].spines['top'].set_visible(False)
    ax[1,2].spines['right'].set_visible(False)

    return(fig)


def partition_large_files(repo):
    data_D_all = pd.read_csv(repo+"/data_D_all.csv")
    data_I_all = pd.read_csv(repo+"/data_I_all.csv")
    data_lambda_all = pd.read_csv(repo+"/data_lambda_all.csv")
    data_R = pd.read_csv(repo+"/data_R.csv")
    data_M = pd.read_csv(repo+"/data_M.csv")
    data_mean_degree_I = pd.read_csv(repo+"/data_mean_degree_I.csv")
    data_I_degrees = pd.read_csv(repo+"/data_I_degrees.csv")
    data_R_lambda = pd.read_csv(repo+"/data_R_lambda.csv")


    k = 0
    for i in [0,100000,200000,300000,400000]:
        data_D_all.loc[i:i+100000-1].to_csv(repo+"/data_D_all_"+str(k)+".csv", index=False)
        data_I_all.loc[i:i+100000-1].to_csv(repo+"/data_I_all_"+str(k)+".csv", index=False)
        data_lambda_all.loc[i:i+100000-1].to_csv(repo+"/data_lambda_all_"+str(k)+".csv", index=False)
        data_R.loc[i:i+100000-1].to_csv(repo+"/data_R_"+str(k)+".csv", index=False)
        data_M.loc[i:i+100000-1].to_csv(repo+"/data_M_"+str(k)+".csv", index=False)
        data_mean_degree_I.loc[i:i+100000-1].to_csv(repo+"/data_mean_degree_I_"+str(k)+".csv", index=False)
        data_I_degrees.loc[i:i+100000-1].to_csv(repo+"/data_I_degrees_"+str(k)+".csv", index=False)
        data_R_lambda.loc[i:i+100000-1].to_csv(repo+"/data_R_lambda_"+str(k)+".csv", index=False)
        k+=1


def apply_ll_large(repo, multiple, LL_berlin):
    frames_D = []
    frames_I = []
    frames_lambda = []
    frames_R = []
    frames_R_lambda = []
    frames_M = []
    frames_mean_degree_I = []
    frames_I_degrees = []

    for k in range(5):
        file = pd.read_csv(repo+"/data_D_all_"+str(k)+".csv")
        print(len(file))
        frames_D.append(file)
        file = pd.read_csv(repo+"/data_I_all_"+str(k)+".csv")
        print(len(file))
        frames_I.append(file)
        file = pd.read_csv(repo+"/data_lambda_all_"+str(k)+".csv")
        frames_lambda.append(file)
        file = pd.read_csv(repo+"/data_R_"+str(k)+".csv")
        frames_R.append(file)
        file = pd.read_csv(repo+"/data_M_"+str(k)+".csv")
        frames_M.append(file)
        file = pd.read_csv(repo+"/data_mean_degree_I_"+str(k)+".csv")
        frames_mean_degree_I.append(file)
        file = pd.read_csv(repo+"/data_I_degrees_"+str(k)+".csv")
        frames_I_degrees.append(file)
        file = pd.read_csv(repo+"/data_R_lambda_"+str(k)+".csv")
        frames_R_lambda.append(file)
        
    data_D_all = pd.concat(frames_D,ignore_index=True)
    data_I_all = pd.concat(frames_I,ignore_index=True)
    data_lambda_all = pd.concat(frames_lambda,ignore_index=True)
    data_R = pd.concat(frames_R,ignore_index=True)
    data_R_lambda = pd.concat(frames_R_lambda,ignore_index=True)
    data_M = pd.concat(frames_M,ignore_index=True)
    data_mean_degree_I = pd.concat(frames_mean_degree_I,ignore_index=True)
    data_I_degrees = pd.concat(frames_I_degrees,ignore_index=True)

    if multiple == np.inf:
        print(len(data_D_all))
        return(data_D_all, data_I_all, data_lambda_all, data_R, data_R_lambda, data_M, data_mean_degree_I, data_I_degrees)    
    else:

        LL_range_D = data_D_all[(data_D_all["LL"] > multiple* LL_berlin)&(data_D_all["LL"] <LL_berlin)]
        LL_range_I = data_I_all[(data_I_all["LL"] > multiple* LL_berlin)&(data_I_all["LL"] < LL_berlin)]
        LL_range_lambda = data_lambda_all[(data_lambda_all["LL"] > multiple* LL_berlin)&(data_lambda_all["LL"] < LL_berlin)]
        LL_range_R = data_R[(data_R["LL"] > multiple* LL_berlin)&(data_R["LL"] < LL_berlin)]
        LL_range_R_lambda = data_R_lambda[(data_R_lambda["LL"] > multiple* LL_berlin)&(data_R_lambda["LL"] < LL_berlin)]
        LL_range_M = data_M[(data_M["LL"] > multiple* LL_berlin)&(data_M["LL"] < LL_berlin)]
        LL_range_mean_degree_I = data_mean_degree_I[(data_mean_degree_I["LL"] > multiple* LL_berlin)&(data_mean_degree_I["LL"] < LL_berlin)]
        LL_range_I_degrees = data_I_degrees[(data_I_degrees["LL"] > multiple* LL_berlin)&(data_I_degrees["LL"] < LL_berlin)]

        print(len(LL_range_D))

        return(LL_range_D, LL_range_I, LL_range_lambda, LL_range_R, LL_range_R_lambda, LL_range_M, LL_range_mean_degree_I, LL_range_I_degrees)



# get a fit for the distribution of contacts

def monoExpb(x, m, t, b):
    return m * np.exp(-t * x) +b

def fit_monoExpb(data_coded, upper_bound, reps = 1, seed = 1):
    data = []
    np.random.seed(seed)
    for i in range(len(data_coded)):
        if data_coded[i] == 6:
            data.append(np.random.randint(5,8, size = reps))
        elif data_coded[i] == 7:
            data.append(np.random.randint(8,11, size = reps))
        elif data_coded[i] == 8:
            data.append(np.random.randint(11,21, size = reps))
        elif data_coded[i] == 9:
            data.append(np.random.randint(21,31, size = reps))
        elif data_coded[i] == 10:
            data.append(np.random.randint(31,41, size = reps))
        elif data_coded[i] == 11:
            data.append(np.random.randint(41,51, size = reps))
        elif data_coded[i] == 12:
            data.append(np.random.randint(51,60, size = reps))
        elif np.isnan(data_coded[i]) == True:
            data.append([0 for x in range(reps)])
        else:
            data.append([data_coded[i] -1 for x in range(reps)])

    xs, counts = np.unique(data, return_counts= True)
    ys = counts/np.sum(counts)
    p0 = (0.25, 1, 0.00001)
    param_bounds=([0,0,0],[np.inf,np.inf, upper_bound])
    params, cv = scipy.optimize.curve_fit(monoExpb, xs, ys, p0, bounds=param_bounds)
    m, t, b = params

    # determine quality of the fit
    squaredDiffs = np.square(ys - monoExpb(xs, m, t, b))
    squaredDiffsFromMean = np.square(ys - np.mean(ys))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"R² = {rSquared}")

    # plot the results
    fig, ax = plt.subplots(1,2, figsize = [8, 4])#[len(cols)*6.4, 4.8])
    ax[0].plot(xs, ys, '.', label="data")
    ax[0].plot(xs, monoExpb(xs, m, t, b), '--', label="fitted")
    ax[0].set_title("Fit for number of contacts")
    ax[1].set_title("CDF")

    cdf = np.cumsum(monoExpb(np.linspace(0,60,61), m, t, b))
    ax[1].step(np.linspace(0,60,61),cdf/cdf[-1])

    # inspect the parameters
    print(f"Y = {m} * e^(-{t} * x) + {b}")

    return(cdf/cdf[-1])


# get a fit for the distribution of contacts

def monoExp(x, m, t):
    return t * np.exp(-t * x)

def fit_monoExp(data_coded, reps = 1, seed = 1):
    data = []
    np.random.seed(seed)
    for i in range(len(data_coded)):
        if data_coded[i] == 6:
            data.append(np.random.randint(5,8, size = reps))
        elif data_coded[i] == 7:
            data.append(np.random.randint(8,11, size = reps))
        elif data_coded[i] == 8:
            data.append(np.random.randint(11,21, size = reps))
        elif data_coded[i] == 9:
            data.append(np.random.randint(21,31, size = reps))
        elif data_coded[i] == 10:
            data.append(np.random.randint(31,41, size = reps))
        elif data_coded[i] == 11:
            data.append(np.random.randint(41,51, size = reps))
        elif data_coded[i] == 12:
            data.append(np.random.randint(51,60, size = reps))
        elif np.isnan(data_coded[i]) == True:
            data.append([0 for x in range(reps)])
        else:
            data.append([data_coded[i] -1 for x in range(reps)])

    xs, counts = np.unique(data, return_counts= True)
    ys = counts/np.sum(counts)
    p0 = (0.25, 1)
    param_bounds=([0,0],[np.inf,np.inf])
    params, cv = scipy.optimize.curve_fit(monoExp, xs, ys, p0, bounds=param_bounds)
    m, t = params

    # determine quality of the fit
    squaredDiffs = np.square(ys - monoExp(xs, m, t))
    squaredDiffsFromMean = np.square(ys - np.mean(ys))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"R² = {rSquared}")

    # plot the results
    fig, ax = plt.subplots(1,2, figsize = [8, 4])#[len(cols)*6.4, 4.8])
    ax[0].plot(xs, ys, '.', label="data")
    ax[0].plot(xs, monoExp(xs, m, t), '--', label="fitted")
    ax[0].set_title("Fit for number of contacts")
    ax[1].set_title("CDF")

    cdf = np.cumsum(monoExp(np.linspace(0,60,61), m, t))
    ax[1].step(np.linspace(0,60,61),cdf/cdf[-1])

    # inspect the parameters
    print(f"Y = {m} * e^(-{t} * x)")

    return(cdf/cdf[-1])



    # create conditinal probs

def get_contact_reduction(data, col, col2):
    contact_red = np.zeros(12)
    values, count = np.unique(data[col], return_counts=True)
    values_reduced, count_reduced = np.unique(data[data[col2] == 1][col], return_counts=True)
    k = 0
    i = 0
    for l in range(len(values_reduced)):
        while (values_reduced[l] != values[k]):
            k += 1

        contact_red[int(values_reduced[l]) - 1] = count_reduced[l]/count[k]

    for i in range(len(contact_red)):
        if not (i+1 in values):
            contact_red[i] = np.nan
           
    return(contact_red)
def square(x, m, b):
    return (m * x**2 + b)

def fit_square(xs, ys):
    p0 = (-1, 1)
    param_bounds=([-np.inf,-np.inf],[np.inf,np.inf])
    params, cv = scipy.optimize.curve_fit(square, xs, ys, p0, bounds=param_bounds)
    m, b = params

    # determine quality of the fit
    squaredDiffs = np.square(ys - square(xs, m, b))
    squaredDiffsFromMean = np.square(ys - np.mean(ys))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"R² = {rSquared}")

    # plot the results
    fig, ax = plt.subplots(1,1, figsize = [4, 4])#[len(cols)*6.4, 4.8])
    ax.plot(xs, ys, '.', label="data")
    ax.plot(xs, square(xs, m, b), '--', label="fitted")
    ax.set_title("Fit for number of contacts of vaccinated")

    # inspect the parameters
    print(f"Y = {m} * x^2 + {b}")

    return(cdf/cdf[-1])

# get lambda plus 
def f(x, unique, counts):
    #unique, counts = np.unique(mu, return_counts=True)
    N = len(unique)
    A = np.zeros((N,N))
    B = np.zeros((N,N))
    for i in range(N):
        A[i,:] = counts
        A[i,i] -= 1
        for j in range(i,N):
            B[i,j] = 1- np.exp(-x[i]*x[j])
            B[j,i] = 1- np.exp(-x[i]*x[j])
    return(np.sum(np.multiply(A,B), axis = 1) - unique)

def get_rates(mu):
    unique, counts = np.unique(mu, return_counts=True)
    guess = np.ones(len(unique))
    sol = optimize.root(f, guess, args = (unique, counts), method='hybr')
    r = sol.x
    
    res_dict = {}
    
    for i in range(len(unique)):
        res_dict[unique[i]] = r[i]
    
    lambda_vec = np.zeros(len(mu))
    for i in range(len(mu)):
        lambda_vec[i] = res_dict[mu[i]]
    
    
    return(lambda_vec)



def combine_results_immune(repo, samples):
    frames_D = []
    frames_I = []
    frames_R0 = []

    for index_start in range(samples):
        index_start = index_start * 100
        index_end = index_start + 100
        try:
            file = repo+"/has_result_D_cum_"+str(index_start)+"_"+str(index_end)+".csv"
            data_D_cum = pd.read_csv(file)
            file = repo+"/has_result_I_cum_"+str(index_start)+"_"+str(index_end)+".csv"
            data_I_cum = pd.read_csv(file)
            file = repo+"/has_result_R0_"+str(index_start)+"_"+str(index_end)+".csv"
            data_R0 = pd.read_csv(file)
            frames_D.append(data_D_cum)
            frames_I.append(data_I_cum)
            frames_R0.append(data_R0)
        except:
            print(index_start)
        

    data_D_all = pd.concat(frames_D)
    data_I_all = pd.concat(frames_I)
    data_R0_all = pd.concat(frames_R0)
    
    return(data_D_all, data_I_all, data_R0_all)
