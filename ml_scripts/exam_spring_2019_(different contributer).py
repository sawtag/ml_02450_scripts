import toolbox_extended as te
import toolbox_02450 as tb
import numpy as np
import pandas as pd
from exam_toolbox import *
import re
import os

import math
import similarity as sim

prep = prep_tools()
pca = pca_calc()
class exam:

    # ----------------------------------------------- OPG 1-----------------------------------------------
    def opg1():
    # it can be seen that x2 has values in [0, 3.7] and x9 in [2.1, 3.7]
        return "D"

    # ----------------------------------------------- OPG 2-----------------------------------------------
    def opg2():
        # We insert the the diagonal numbers from the S matrix
        diag_values = [14.14, 11.41, 9.46, 4.19, 0.17]
        pca.draw_curve_from_diagonal_values(diag_values)
        # Can also be done by:
        pca.var_explained(diag_values)
        
        # A) k=2 -> explains 0.755
        # B) k=1 -> explains 0.45
        # C) last 4 -> ~0.54 < 0.56 
        last_4 = 1 - 0.457314
        return "C"

    # ----------------------------------------------- OPG 3-----------------------------------------------
    def opg3():
        # A) PCA1 -> low museums (0.94) high religious (-0.33) -> negative value
        return "A"

    # ----------------------------------------------- OPG 4-----------------------------------------------
    def opg4():
        # According to text on the end of page 331 in the book, the answer is B
    
        # Add the k closest distances from ox
        o7 = np.array([2.2, 1.7])
        o6 = np.array([1.8, 1.7])
        o4 = np.array([0.9, 2.1])
        
        den = round(te.density(o7) / np.mean([te.density(o6), te.density(o4)]), 2)
        
        print(den)
        return "D"

    # ----------------------------------------------- OPG 5-----------------------------------------------
    def opg5():
        # o1 -> C2 -> wrong
        # o2 -> C2 -> wrong
        # o3 -> C3 -> wrong
        # ...
        # 6 wrong in 10
        return "C"

    # ----------------------------------------------- OPG 6-----------------------------------------------
    def opg6():
        # o6 - o4 distance is 2.1 -> exclude Dendograms 3 and 4
        # [o3, o9] - o8 max distance is 5.4 -> exclude Dendogram 1
        return "B"

    # ----------------------------------------------- OPG 7-----------------------------------------------
    def opg7():
        f11 = 4 #{(1,2), (4,5), (6,7), (8,9)}
        f00 = 17
        N = 10
        M = N*(N-1)/2
        J = f11 / (M - f00)
        print(J) # ~0.143
        return "B"

    # ----------------------------------------------- OPG 8-----------------------------------------------
    def opg8():
        tree = decision_trees()
        root = [263, 359, 358]
        div_0_43 = [[143, 137, 54],[120, 222, 304]]
        tree.purity_gain(root,div_0_43[0], div_0_43[1], "class_error")
        # I(<=0.43) =~ 0.0898
        return "B"

    # ----------------------------------------------- OPG 9-----------------------------------------------
    def opg9():
        div_0_55 = [[223, 251, 197],[40, 108, 161]]
        
        print((max(div_0_55[0]) + max(div_0_55[1])) / np.sum(div_0_55))
        # ~0.42
        return "A"

    # ----------------------------------------------- OPG 10-----------------------------------------------
    def opg10():
        # calculate edge values 
        # max = w0 + max(w(2)*h(1))[0] + max(w(2)*h(1))[1]
        # 2.2 + (-0.3 * 0) + (0.5 * 1) = 2.7
        # min = w0 + min(w(2)*h(1))[0] + min(w(2)*h(1))[1]
        # 2.2 + (-0.3 * 1) + (0.5 * 0) = 1.9
        
        # Output 4 has limits [2, 2.6]
        return "D"

    # ----------------------------------------------- OPG 11-----------------------------------------------
    def opg11():
        # consider the position (0, -1)
        test_p = [1, 0, -1]
        
        # A
        w1 = np.array([-0.77, -5.54, 0.01])
        w2 = np.array([0.26, -2.09, -0.03])
        
        y1 = np.sum(w1.T * test_p) # -0.78
        y2 = np.sum(w2.T * test_p) # 0.29
        # -> class 2
        
        # B
        w1 = np.array([0.51, 1.65, 0.01])
        w2 = np.array([0.1, 3.8, 0.04])
        
        y1 = np.sum(w1.T * test_p) # 0.5
        y2 = np.sum(w2.T * test_p) # 0.06
        # -> class 1
        
        # C
        w1 = np.array([-0.9, -4.39, 0])
        w2 = np.array([-0.09, -2.45, -0.04])
        
        y1 = np.sum(w1.T * test_p) # -0.9
        y2 = np.sum(w2.T * test_p) # -0.05
        # -> class 3 (0 is bigger than both the other classes)
        
        
        # D
        w1 = np.array([-1.22, -9.88, -0.01])
        w2 = np.array([-0.28, -2.9, -0.01])
        
        y1 = np.sum(w1.T * test_p) # -1.21
        y2 = np.sum(w2.T * test_p) # -0.27
        # -> class 3 (0 is bigger than both the other classes)
        
        # Only A weights predict class 2, which is the correct one
        return "A"

    # ----------------------------------------------- OPG 12-----------------------------------------------
    def opg12():
        observations = [1.0, 1.2, 1.8, 2.3, 2.6, 3.4, 4, 4.1, 4.2, 4.6]
        clus = cluster()
        clus.kmeans_1d(observations,3,init=[1.8, 3.3, 3.6])
        # converged = [1.3333 2.7667 4.225 ]
        return "C"

    # ----------------------------------------------- OPG 13-----------------------------------------------
    def opg13():
        
        # We add the binary table
        data =[[0, 1, 0],
               [0, 0, 0],
               [1, 1, 1],
               [0, 0, 0],
               [0, 1, 0],
               [0, 1, 0],
               [0, 1, 1],
               [0, 0, 1],
               [1, 0, 1],
               [0, 1, 1]]
        df = pd.DataFrame(data)
        super = supervised()
        super.naive_bayes([0, 0, 1, 1, 1, 2, 2, 2, 2, 2],df,[0, 1, 2],[0, 1, 0], 1)
        # We see the answer is 0.37523 ~= 200/533
        
        return "A"

    # ----------------------------------------------- OPG 14-----------------------------------------------
    def opg14():
        # support > 0.15 = at least 2 occurrences
        
        # {f2} support > 0.15 -> we exclude B and C
        # {f2, f3} support > 0.15 -> we exclude D
        return "A"

    # ----------------------------------------------- OPG 15-----------------------------------------------
    def opg15():
        # confidence a -> b
        # supp(a) / supp(a, b)
        
        # supp({f2}) = 2/10
        # supp({f2, f3, f4, f5, f6}) = 1/10
        
        # conf = 1/2
        
        return "B"

    # ----------------------------------------------- OPG 16-----------------------------------------------
    def opg16():
        # as we see in L1, only f3 and f4 are frequent sets
        # therefore, the only candidate set for C2 is {f3, f4}
        return "B"

    # ----------------------------------------------- OPG 17-----------------------------------------------
    def opg17():
        o1 = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        o2 = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        o3 = [0, 1, 1, 1, 1, 1, 0, 0, 0]
        o4 = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        # cos o1,o3 = 0.44
        # J o2,o3 = 0.0
        # smc o1,o3 = 0.55
        # smc o2,o4 = 0.77
        similarity().measures(o2, o4)
        return "B"

    # ----------------------------------------------- OPG 18-----------------------------------------------
    def opg18():
        true=[1, 1, 0, 1, 1, 1, 0]
        pred=[.14, .15, 0.27, .61, .71, .75, .81]
        tb.rocplot(pred, true)
        return "D"

    # ----------------------------------------------- OPG 19-----------------------------------------------
    def opg19():
        # Forward Selection
        # f=1 -> x6 -> 4.57
        # f=2 -> x1, x6 -> 4.213
        # f=3 -> x1, x6, x7 -> 4.161
        # f=4 -> x1, x6, x7, x8 -> 4.098
        # f=5 -> worse than f=4
        return "B"

    # ----------------------------------------------- OPG 20-----------------------------------------------
    def opg20():
        # p(y1|x2=0, x3=1) =
        
        #         p(x2=0|y1) . p(x3=1|y1) . p(y1)
        # _________________________________________________
        # sum, for i in [1,2,3] p(x2=0|yi).p(x3=1|yi).p(yi)
        
        # = ~0.17
        return "A"

    # ----------------------------------------------- OPG 21-----------------------------------------------
    def opg21():
        
        pos = np.array([5, 1])
        
        rA = np.array([2, 4])
        rB = np.array([6, 0])
        rC = np.array([4, 2])
        
        
        A = np.linalg.norm(pos - rA, ord=1) < 2
        B = np.linalg.norm(pos - rB, ord=2) < 3
        C = np.linalg.norm(pos - rC, ord=2) < 2
        print(A, B, C)
        
        return "A"

    # ----------------------------------------------- OPG 22-----------------------------------------------
    def opg22():
        
        outer = 4
        inner = 5
        parameters = 5
        
        print(outer * (inner*parameters + 1))
        # 104 * 2 (ANN + lin reg)
        return "A"

    # ----------------------------------------------- OPG 23-----------------------------------------------
    def opg23():
        weights = [0.19,0.34,0.48]
        mus = [3.177,3.181,3.184]
        sigmas = [0.0062,0.0076,0.0075] #sigmas are NOT squared
        xpredict = 3.19
        
        #which cluster to predict
        K = 2
        
        def N(x, mu, sigma):
            bottom = math.sqrt(2*math.pi*(sigma**2))
            # bottom = math.sqrt(2*math.pi*sigma)
            topE = (x-mu)**2
            botE = 2*(sigma**2)
            expIn = -1*(topE/botE)
            exp = math.e**(expIn)
            return exp/bottom
        
        def p(x):
            tot=0
            for i in range(0,len(weights)):
                n = N(x,mus[i],sigmas[i])
                tot = tot + weights[i]*n
            return tot
        
        #Probability bk of xpredict being in cluser K
        bk = N(xpredict, mus[K-1],sigmas[K-1])*weights[K-1]/p(xpredict)
        # ~0.31
        return "B"

    # ----------------------------------------------- OPG 24-----------------------------------------------
    def opg24():
        miss = np.zeros(7) 
        miss[:2] = 1 
        te.adaboost(miss, rounds=1)
        return "A"

    # ----------------------------------------------- OPG 25-----------------------------------------------
    def opg25():
        # TODO
        return "E" # :'(

    # ----------------------------------------------- OPG 26-----------------------------------------------
    def opg26():
        # by analyzing the covariance matrixes, we can infer which one is correct
        # by looking and the cov(x1, x2), which is in position [0][1]
        # we can see that matrix 2 has a negative value, while the plot shows a positive correlation
        
        # so we know that matrix 1 is valid
        # then, to calculate the correlation we have:
        # corr(x, y) = cov(x, y) / sqrt(sd(x) * sd(y))
        cov = 0.56
        sdX1 = 0.5
        sdX2 = 1.5
        
        print(cov / math.sqrt(sdX1 * sdX2))
        # ~0.647
        return "C"

    # ----------------------------------------------- OPG 27-----------------------------------------------
    def opg27():
        # taking the cluster with mean [-6.8, 6.4]
        # it's easy to see that the only valid option is A
        # as all the others have a negative correlation between x1 and x2

        return "A"

    # -------------------------------- answers dataframe -------------------------------------------------
    def answers(show=True, csv=False, excel=False):
        ans = pd.DataFrame(
            columns=["Student number: s123456"]
        )  # columns = ["OPG", "svar"])

#        ans.loc[0] = ""
#        ans.loc[1] = "Q01: {}".format(exam.opg1())
#        ans.loc[2] = "Q02: {}".format(exam.opg2())
#        ans.loc[3] = "Q03: {}".format(exam.opg3())
#        ans.loc[4] = "Q04: {}".format(exam.opg4())
#        ans.loc[5] = "Q05: {}".format(exam.opg5())
#        ans.loc[6] = "Q06: {}".format(exam.opg6())
#        ans.loc[7] = "Q07: {}".format(exam.opg7())
#        ans.loc[8] = "Q08: {}".format(exam.opg8())
#        ans.loc[9] = "Q09: {}".format(exam.opg9())
#        ans.loc[10] = "Q10: {}".format(exam.opg10())
#        ans.loc[11] = ""

#        ans.loc[12] = "Q11: {}".format(exam.opg11())
#        ans.loc[13] = "Q12: {}".format(exam.opg12())
#        ans.loc[14] = "Q13: {}".format(exam.opg13())
#        ans.loc[15] = "Q14: {}".format(exam.opg14())
#        ans.loc[16] = "Q15: {}".format(exam.opg15())
#        ans.loc[17] = "Q16: {}".format(exam.opg16())
#        ans.loc[18] = "Q17: {}".format(exam.opg17())
#        ans.loc[19] = "Q18: {}".format(exam.opg18())
#        ans.loc[20] = "Q19: {}".format(exam.opg19())
#        ans.loc[21] = "Q20: {}".format(exam.opg20())
#        ans.loc[22] = ""

#        ans.loc[23] = "Q21: {}".format(exam.opg21())
#        ans.loc[24] = "Q22: {}".format(exam.opg22())
#        ans.loc[25] = "Q23: {}".format(exam.opg23())
#        ans.loc[26] = "Q24: {}".format(exam.opg24())
#        ans.loc[27] = "Q25: {}".format(exam.opg25())
#        ans.loc[28] = "Q26: {}".format(exam.opg26())
#        ans.loc[29] = "Q27: {}".format(exam.opg27())

        if excel:
            ans.to_excel(re.sub(".py", "_answers.xlsx", __file__), index=False)
        if csv:
            ans.to_csv(re.sub(".py", "_answers.csv", __file__), index=False)
        if show:
            print(ans)

        return ans


exam.answers()
