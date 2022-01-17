import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
env = 'pusher'

#pusher
#'''''
df_gcsl_0 = pd.read_csv('data/example/door/gcsl_sto_nb_off 0/2022_01_15_18_14_27/progress.csv')
df_gcsl_1 = pd.read_csv('/home/nsh1609/gcsl-norm/data/example/pusher/gcsl_o_1/2022_01_09_12_17_55/progress.csv')
df_gcsl_2 = pd.read_csv('data/example/pusher/gcsl_sto_off_ 2/2022_01_13_00_44_33/progress.csv')
df_gcsl_3 = pd.read_csv('data/example/pusher/gcsl_sto_off_ 3/2022_01_13_00_44_33/progress.csv')
df_gcsl_4 = pd.read_csv('data/example/pusher/gcsl_sto_off_ 4/2022_01_13_00_44_33/progress.csv')
df_gcsl_5 = pd.read_csv('data/example/pusher/gcsl_sto_off_ 5/2022_01_13_00_44_33/progress.csv')


time_0 = df_gcsl_0['timesteps'].values
success_0 = df_gcsl_0['Eval success ratio'].values
time_1 = df_gcsl_1['timesteps'].values
success_1 = df_gcsl_1['Eval success ratio'].values
time_2 = df_gcsl_2['timesteps'].values
success_2 = df_gcsl_2['Eval success ratio'].values
time_3 = df_gcsl_3['timesteps'].values
success_3 = df_gcsl_3['Eval success ratio'].values
time_4 = df_gcsl_4['timesteps'].values
success_4 = df_gcsl_4['Eval success ratio'].values
time_5 = df_gcsl_5['timesteps'].values
success_5 = df_gcsl_5['Eval success ratio'].values
#'''''

#'''''
n_df_gcsl_0 = pd.read_csv('/home/nsh1609/gcsl-norm/data/example/door/gcsl_mt2_sto_1/2022_01_08_10_53_34/progress.csv')
n_df_gcsl_1 = pd.read_csv('data/example/door/ne_sto_nobias_onp0/2022_01_16_15_43_08/progress.csv')
n_df_gcsl_2 = pd.read_csv('data/example/door/norm_sto_nb1_on 0/2022_01_15_11_42_26/progress.csv')
n_df_gcsl_3 = pd.read_csv('data/example/pusher/norm_sto_ 3/2022_01_13_08_22_22/progress.csv')
n_df_gcsl_4 = pd.read_csv('data/example/pusher/norm_sto_ 4/2022_01_13_08_22_22/progress.csv')
n_df_gcsl_5 = pd.read_csv('data/example/pusher/norm_sto_ 5/2022_01_13_08_22_22/progress.csv')


n_time_0 = n_df_gcsl_0['timesteps'].values
n_success_0 = n_df_gcsl_0['Eval success ratio'].values
n_time_1 = n_df_gcsl_1['Validation loss'].values
n_success_1 = n_df_gcsl_1['Eval success ratio'].values
n_time_2 = n_df_gcsl_2['Validation loss'].values
n_success_2 = n_df_gcsl_2['Eval success ratio'].values
n_time_3 = n_df_gcsl_3['Validation loss'].values
n_success_3 = n_df_gcsl_3['Eval success ratio'].values
n_time_4 = n_df_gcsl_4['Validation loss'].values
n_success_4 = n_df_gcsl_4['Eval success ratio'].values
n_time_5 = n_df_gcsl_5['Validation loss'].values
n_success_5 = n_df_gcsl_5['Eval success ratio'].values
#'''''

#s_l = df_gcsl_0['Eval success ratio_left'].values
#s_r = df_gcsl_0['Eval success ratio_right'].values

#n_s_l = n_df_gcsl_0['Eval success ratio_left'].values
#n_s_r = n_df_gcsl_0['Eval success ratio_right'].values
#plt.plot(time_0,success_0,'b', label = 'M0')
#plt.plot(time_1,success_1,'r',label = 'M1')
#plt.plot(time_2,success_2,'g',label = 'M2')
#plt.plot(time_3,success_3,'k',label = 'M3')
#plt.plot(time_4,success_4,'m',label = 'M4')
#plt.plot(time_5,success_5,'y',label = 'M5')

plt.plot(n_time_0,n_success_0,'b', label = 'M0')
plt.plot(n_time_1,n_success_1,'r', label = 'N_M0')
#plt.plot(time_1,success_1,'g', label = 'N_M0')

plt.xlabel('TimeSteps')
plt.ylabel('Success')
plt.legend()
plt.title('Success - M0  -' + env )
plt.grid()
plt.show()
#plt.savefig('plots_new/'+env+'_loss/Success.jpg')
plt.close()






