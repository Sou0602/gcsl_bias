import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
env = 'lunar'

#pusher
#'''''
df_gcsl_0 = pd.read_csv('data/example/pusher/gcsl_offline_s0/2022_01_17_00_39_05/progress.csv')
df_gcsl_1 = pd.read_csv('data/example/pointmass_rooms/gcsl_offpb_s2_0/2022_01_24_17_19_43/progress.csv')
df_gcsl_2 = pd.read_csv('data/example/lunar/gcsl_offpb_s3_3/2022_01_24_16_15_04/progress.csv')
df_gcsl_3 = pd.read_csv('example/pointmass_rooms/gcsl_offpb_s0_1/2022_01_24_13_30_46/progress.csv')
df_gcsl_4 = pd.read_csv('example/pusher/gcsl_offpb_s5_2/2022_01_24_07_37_22/progress.csv')
df_gcsl_5 = pd.read_csv('data/example/pusher/gcsl_offp_s5/2022_01_18_01_40_00/progress.csv')


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
n_df_gcsl_0 = pd.read_csv('data/example/pusher/norm_offline_s0/2022_01_17_10_32_02/progress.csv')
n_df_gcsl_1 = pd.read_csv('data/example/pointmass_rooms/norm_offpb_s2_0/2022_01_24_17_19_43/progress.csv')
n_df_gcsl_2 = pd.read_csv('data/example/lunar/norm_offpb_s3_3/2022_01_24_16_15_03/progress.csv')
n_df_gcsl_3 = pd.read_csv('example/pointmass_rooms/norm_offpb_s0_1/2022_01_24_13_30_46/progress.csv')
n_df_gcsl_4 = pd.read_csv('example/pusher/norm_offpb_s5_2/2022_01_24_07_37_22/progress.csv')
n_df_gcsl_5 = pd.read_csv('data/example/pusher/norm_offp_s5/2022_01_18_01_40_00/progress.csv')


n_time_0 = n_df_gcsl_0['Validation loss'].values
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

plt.plot(time_1,success_1,'b', label = 'M0')
plt.plot(n_time_1,n_success_1,'r', label = 'N_M0')
#plt.plot(time_1,success_1,'g', label = 'N_M0')

plt.xlabel('TimeSteps')
plt.ylabel('Success')
plt.legend()
plt.title('Success - M5  -' + env )
plt.grid()
plt.show()
#plt.savefig('plots_new/'+env+'_loss/Success.jpg')
plt.close()






