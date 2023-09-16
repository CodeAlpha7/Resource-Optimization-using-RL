import cvxpy as cp
import matplotlib.pyplot as matplt
from utils import *
from tst_ddpg import *
from ddpg_alg_spinup import ddpg
import tensorflow as tf
from env_mra import ResourceEnv
import numpy as np
import time
import pickle
import scipy.io
from parameters import *
from functions import *
import multiprocessing
from numpy.random import default_rng




def admm_static_algorithm(sliceNo, ueNo, resNo, alpha, weight, ):
    # static allocation
    real_utility_static = np.zeros(sliceNo)

    for i in range(sliceNo):

        tmp_utility, tmp_real_utility, x = np.zeros(resNo), np.zeros(resNo), np.zeros([sliceNo, resNo, ueNo],
                                                                                      dtype=np.float32)

        for j in range(resNo):
            tmp_utility[j], x[i, j], tmp_real_utility[j] = \
                simple_static_alogrithm(z_minus_u=Rmax / sliceNo,
                                        alpha=alpha[i, j],
                                        weight=weight[i],
                                        UENum=ueNo,
                                        minReward=minReward / maxTime)

        real_utility_static[i] = np.mean(tmp_real_utility) * maxTime

    real_utility_static = np.sum(real_utility_static)
    return real_utility_static, 0
    #  static allocation


def admm_ddpg_algorithm(sliceNo, ueNo, resNo, alpha, weight, idx):
    # ddpg allocation

    z = np.zeros([sliceNo, resNo], dtype=np.float32)
    u = np.zeros([sliceNo, resNo], dtype=np.float32)
    x = np.zeros([sliceNo, resNo, ueNo], dtype=np.float32)
    z_minus_u = z - u

    sum_utility, sum_real_utility, sum_gap, sum_x = [], [], [], []

    for ite in range(ADMM_iter):
        aug_utility = np.zeros(sliceNo)
        real_utility = np.zeros(sliceNo)

        # x-update in each slice #####################################
        for i in range(sliceNo):
            aug_utility[i], tmpx, real_utility[i] = load_and_run_policy(agent_id=idx[i],
                                                                        alpha=alpha[i],
                                                                        weight=weight[i],
                                                                        UENum=ueNo,
                                                                        RESNum=resNo,
                                                                        aug_penalty=z_minus_u[i])
            print('tmpx is')
            print(tmpx)
            x[i] = Rmax * np.mean(tmpx, axis=0)  # mean for all maxTime

        sumx = np.sum(x, axis=2)  # the sum resource of all users in each slice

        for j in range(resNo):
            # z-update #####################################
            z[:, j] = optimize_z_function(sumx=sumx[:, j], u=u[:, j], SliceNum=sliceNo)

        # u-update #####################################
        u = u + (sumx - z)

        z_minus_u = np.clip(z - u, Rmin, Rmax)  # in case of overflow or negative

        sum_utility.append(np.sum(aug_utility))
        sum_real_utility.append(np.sum(real_utility))
        sum_gap.append(np.abs(np.mean((np.sum(sumx, axis=0) - Rmax) / Rmax)))
        sum_x.append(sumx)

        print('iter ' + str(ite))
        print('current allocation is')
        print(sumx)

    return sum_real_utility, sum_gap
    #  ddpg allocation

def admm_td3_algorithm(sliceNo, ueNo, resNo, alpha, weight, idx):
    # td3 allocation

    z = np.zeros([sliceNo, resNo], dtype=np.float32)
    u = np.zeros([sliceNo, resNo], dtype=np.float32)
    x = np.zeros([sliceNo, resNo, ueNo], dtype=np.float32)
    z_minus_u = z - u

    sum_utility, sum_real_utility, sum_gap, sum_x = [], [], [], []

    for ite in range(ADMM_iter):
        aug_utility = np.zeros(sliceNo)
        real_utility = np.zeros(sliceNo)

        # x-update in each slice #####################################
        for i in range(sliceNo):
            aug_utility[i], tmpx, real_utility[i] = load_and_run_policy(agent_id=idx[i],
                                                                        alpha=alpha[i],
                                                                        weight=weight[i],
                                                                        UENum=ueNo,
                                                                        RESNum=resNo,
                                                                        aug_penalty=z_minus_u[i])

            x[i] = Rmax * np.mean(tmpx, axis=0)  # mean for all maxTime

        sumx = np.sum(x, axis=2)  # the sum resource of all users in each slice

        for j in range(resNo):
            # z-update #####################################
            z[:, j] = optimize_z_function(sumx=sumx[:, j], u=u[:, j], SliceNum=sliceNo)

        # u-update #####################################
        u = u + (sumx - z)

        z_minus_u = np.clip(z - u, Rmin, Rmax)  # in case of overflow or negative

        sum_utility.append(np.sum(aug_utility))
        sum_real_utility.append(np.sum(real_utility))
        sum_gap.append(np.abs(np.mean((np.sum(sumx, axis=0) - Rmax) / Rmax)))
        sum_x.append(sumx)

        print('iter ' + str(ite))
        print('current allocation is')
        print(sumx)

    return sum_real_utility, sum_gap
    #  ddpg allocation


def admm_opt_algorithm(sliceno, ueno, resno, alpha, weight):
    #  ddpg allocation

    z = np.zeros([sliceno, resno], dtype=np.float32)
    u = np.zeros([sliceno, resno], dtype=np.float32)
    x = np.zeros([sliceno, resno, ueno], dtype=np.float32)
    z_minus_u = z - u

    sum_utility, sum_real_utility, sum_gap, sum_x = [], [], [], []

    for ite in range(ADMM_iter):
        aug_utility = np.zeros(sliceno)
        real_utility = np.zeros(sliceno)

        # x-update in each slice #####################################
        for i in range(sliceno):

            tmp_utility, tmp_real_utility = np.zeros(resno), np.zeros(resno)

            for j in range(resno):
                # since all the conditions are the same for all time slots, we assign the same results to all time slots
                tmp_utility[j], x[i, j], tmp_real_utility[j] = \
                    simple_convex_alogrithm(z_minus_u=z_minus_u[i, j],
                                            alpha=alpha[i, j],
                                            weight=weight[i],
                                            UENum=ueno,
                                            minReward=minReward / maxTime)

            aug_utility[i] = np.mean(tmp_utility) * maxTime  # utility of slice -- mean for all resources

            real_utility[i] = np.mean(tmp_real_utility) * maxTime  # utility of slice -- mean for all resources

        sumx = np.sum(x, axis=2)  # the sum resource of all users in each slice

        for j in range(resno):
            # z-update #####################################
            z[:, j] = optimize_z_function(sumx=sumx[:, j], u=u[:, j], SliceNum=sliceno)

        # u-update #####################################
        u = u + (sumx - z)

        z_minus_u = np.clip(z - u, Rmin, Rmax)  # in case of overflow or negative

        sum_utility.append(np.sum(aug_utility))
        sum_real_utility.append(np.sum(real_utility))
        sum_gap.append(np.abs(np.mean((np.sum(sumx, axis=0) - Rmax) / Rmax)))
        sum_x.append(sumx)

        print('iter ' + str(ite))
        print('current allocation is')
        print(sumx)

    return sum_real_utility, sum_gap
    #  ddpg allocation


def admm_mix_algorithm(SliceNum, UENum, RESNum, alpha, weight, simulated_optimization):
    #  ddpg allocation

    z = np.zeros([SliceNum, RESNum], dtype=np.float32)
    u = np.zeros([SliceNum, RESNum], dtype=np.float32)
    x = np.zeros([SliceNum, RESNum, UENum], dtype=np.float32)
    z_minus_u = z - u

    sum_utility, sum_real_utility, sum_gap, sum_x = [], [], [], []

    for ite in range(ADMM_iter):
        aug_utility = np.zeros(SliceNum)
        real_utility = np.zeros(SliceNum)

        # x-update in each slice #####################################
        for i in range(SliceNum):

            if simulated_optimization[i]:

                tmp_utility, tmp_real_utility = np.zeros(RESNum), np.zeros(RESNum)

                for j in range(RESNum):
                    # since all the conditions are the same for all time slots, we assign the same results to all time slots
                    tmp_utility[j], x[i, j], tmp_real_utility[j] = \
                        simple_convex_alogrithm(z_minus_u=z_minus_u[i, j],
                                                alpha=alpha[i, j],
                                                weight=weight[i],
                                                UENum=UENum,
                                                minReward=minReward / maxTime)

                aug_utility[i] = np.mean(tmp_utility) * maxTime  # utility of slice -- mean for all resources

                real_utility[i] = np.mean(tmp_real_utility) * maxTime  # utility of slice -- mean for all resources

            else:

                aug_utility[i], tmpx, real_utility[i] = load_and_run_policy(agent_id=i,
                                                                            alpha=alpha[i],
                                                                            weight=weight[i],
                                                                            UENum=UENum,
                                                                            RESNum=RESNum,
                                                                            aug_penalty=z_minus_u[i])

                x[i] = Rmax * np.mean(tmpx, axis=0)  # mean for all maxTime

        sumx = np.sum(x, axis=2)  # the sum resource of all users in each slice

        for j in range(RESNum):
            # z-update #####################################
            z[:, j] = optimize_z_function(sumx=sumx[:, j], u=u[:, j], SliceNum=SliceNum)

        # u-update #####################################
        u = u + (sumx - z)

        z_minus_u = np.clip(z - u, Rmin, Rmax)  # in case of overflow or negative

        sum_utility.append(np.sum(aug_utility))
        sum_real_utility.append(np.sum(real_utility))
        sum_gap.append(np.abs(np.mean((np.sum(sumx, axis=0) - Rmax) / Rmax)))
        sum_x.append(sumx)

        print('iter ' + str(ite))
        print('current allocation is')
        print(sumx)

    return sum_real_utility
    # ddpg allocation


def main_admm_algorithm(SliceNum, UENum, RESNum, alpha, weight, INDEX):
    utility_static = admm_static_algorithm(sliceNo=SliceNum, ueNo=UENum, resNo=RESNum, alpha=alpha, weight=weight, )

    utility_ddpg = admm_ddpg_algorithm(sliceNo=SliceNum, ueNo=UENum, resNo=RESNum, alpha=alpha, weight=weight, idx=INDEX)[-1]

    utility_opt = admm_opt_algorithm(sliceno=SliceNum, ueno=UENum, resno=RESNum, alpha=alpha, weight=weight, )[-1]

    # utility_td3 = admm_td3_algorithm(sliceNo=SliceNum, ueNo=UENum, resNo=RESNum, alpha=alpha, weight=weight, idx=INDEX)[-1]
    print("APPY")
    print([utility_static, utility_ddpg, utility_opt])

    # scipy.io.savemat('/root/Desktop/DRL_Project/ADMM_simulation' + str(simulated_optimization) + '.mat', mdict={'sum_x': sum_x,
    #                                                'sum_real_utility': sum_real_utility,
    #                                                'sum_gap': sum_gap,
    #                                                'sum_utility': sum_utility,
    #                                                'real_utility_static':real_utility_static,})

    return utility_static, utility_ddpg, utility_opt


if __name__ == "__main__":

    rng = default_rng()
    org_alpha = alpha
    org_weight = weight

    V = np.arange(SliceNum)

    Times_Vec = [20] * SliceNum


    for iSlice in range(len(Times_Vec)):

        print(iSlice)
        Times = Times_Vec[iSlice]
        sliceno = iSlice
        print("sliceno", sliceno)

        iseed = 0

        utility_static, utility_ddpg, utility_opt = np.zeros(Times), np.zeros(Times), np.zeros(Times)

        for ite in range(Times):
            print(ite)
            iseed += 4565
            np.random.seed(iseed)

            INDEX = rng.choice(V, SliceNum, replace=False)
            print("idx", INDEX)
            alpha_ = org_alpha[INDEX]
            weight_ = org_weight[INDEX]

            print('INDEX IS ' + str(INDEX))

            utility_static[ite], utility_ddpg[ite], utility_opt[ite] = main_admm_algorithm(SliceNum, UENum, RESNum,
                                                                                           alpha_, weight_, INDEX)

        scipy.io.savemat('/Users/apoorvgarg/PycharmProjects/BTP-Slice-RL/result/Apooorv_tests' + str(SliceNum) + '.mat',
                         mdict={'utility_static': utility_static,
                                'utility_ddpg': utility_ddpg,
                                'utility_opt': utility_opt, })
