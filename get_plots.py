import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
env = 'door'
#'''''
#door
df_gcsl_nt = pd.read_csv('/home/nsh1609/gcsl-norm/data/example/door/gcsl_mt2/2022_01_04_10_18_54/progress.csv')
df_gcsl_n = pd.read_csv('data/example/door/norm_sto_off/2022_01_12_13_10_37/progress.csv')
df_gcsl = pd.read_csv('data/example/door/gcsl_sto_off/2022_01_12_11_52_14/progress.csv')
#df_gcsl_s = pd.read_csv('data/example/door/gcsl/2021_12_19_00_55_52/progress.csv')
#'''''
'''''
#pusher
df_gcsl_nt = pd.read_csv('/home/nsh1609/gcsl-norm/data/example/pusher/gcsl_mt2/2022_01_04_10_18_53/progress.csv')
df_gcsl_n = pd.read_csv('data/example/pusher/norm_sto_off/2022_01_12_13_10_36/progress.csv')
df_gcsl = pd.read_csv('data/example/pusher/gcsl_sto_off/2022_01_12_11_52_14/progress.csv')
'''''
'''''
#lunar
df_gcsl_nt = pd.read_csv('data/example/lunar/gcsl_mt2/2022_01_04_09_07_39/progress.csv')
df_gcsl_n = pd.read_csv('data/example/lunar/gcsl_mt/2021_12_23_13_30_35/progress.csv')
df_gcsl = pd.read_csv('data/example/lunar/gcsl_od/2021_12_24_09_55_32/progress.csv')
'''''
'''''
#pointmass_rooms
df_gcsl_nt = pd.read_csv('data/example/pointmass_rooms/gcsl_mt2/2022_01_04_09_07_39/progress.csv')
df_gcsl_n = pd.read_csv('data/example/pointmass_rooms/gcsl_mt/2021_12_23_12_53_03/progress.csv')
df_gcsl = pd.read_csv('data/example/pointmass_rooms/gcsl_od/2021_12_24_09_55_32/progress.csv')
'''''
'''''
#pointmass_empty
df_gcsl_nt = pd.read_csv('data/example/pointmass_empty/gcsl_mt2_sto_1/2022_01_07_15_00_53/progress.csv')
df_gcsl_n = pd.read_csv('data/example/pointmass_empty/gcsl_mt2_sto_1/2022_01_06_08_57_07/progress.csv')
df_gcsl = pd.read_csv('data/example/pointmass_empty/gcsl_o/2021_12_24_01_24_31/progress.csv')
'''''

nt_time = df_gcsl_nt['timesteps'].values
nt_success = df_gcsl_nt['Eval success ratio'].values
nt_avg_dist = df_gcsl_nt['Eval avg final dist'].values
nt_loss = df_gcsl_nt['policy loss'].values

n_time = df_gcsl_n['timesteps'].values
n_success = df_gcsl_n['Eval success ratio'].values
n_avg_dist = df_gcsl_n['Eval avg final dist'].values
n_loss = df_gcsl_n['policy loss'].values
n_success_l = df_gcsl_n['Eval success ratio_left']
n_success_r = df_gcsl_n['Eval success ratio_right']
c_loss = df_gcsl_n['Conditional loss']
m_loss = df_gcsl_n['Marginal loss']


#n_dist_mean = df_gcsl_n['Eval final puck distance Mean'].values
#n_dist_std = df_gcsl_n['Eval final puck distance Std'].values

time = df_gcsl['timesteps'].values
success = df_gcsl['Eval success ratio'].values
avg_dist = df_gcsl['Eval avg final dist'].values
loss = df_gcsl['policy loss'].values
success_l = df_gcsl['Eval success ratio_left']
success_r = df_gcsl['Eval success ratio_right']
#dist_mean = df_gcsl['Eval final puck distance Mean'].values
#dist_std = df_gcsl['Eval final puck distance Std'].values
## Plot 1
#print(time)
plt.plot(time,success,'b', label = 'GCSL')
plt.plot(time,n_success,'r',label = 'Norm')
#plt.plot(nt_time,nt_success,'k',label = 'GCSL_Norm_T_loss+')
plt.xlabel('TimeSteps')
plt.ylabel('Success')
plt.legend()
plt.title('Success -' + env )
plt.grid()
plt.show()
#plt.savefig('plots_new/'+env+'_loss/Success.jpg')
plt.close()

plt.plot(time,success_l,'b', label = 'GCSL')
plt.plot(time,n_success_l,'r',label = 'Norm')
#plt.plot(nt_time,nt_success,'k',label = 'GCSL_Norm_T_loss+')
plt.xlabel('TimeSteps')
plt.ylabel('Success')
plt.legend()
plt.title('Success-Left -' + env )
plt.grid()
plt.show()
#plt.savefig('plots_new/'+env+'_loss/Success.jpg')
plt.close()

plt.plot(time,success_r,'b', label = 'GCSL')
plt.plot(time,n_success_r,'r',label = 'Norm')
#plt.plot(nt_time,nt_success,'k',label = 'GCSL_Norm_T_loss+')
plt.xlabel('TimeSteps')
plt.ylabel('Success')
plt.legend()
plt.title('Success-Right -' + env )
plt.grid()
plt.show()
#plt.savefig('plots_new/'+env+'_loss/Success.jpg')
plt.close()

plt.plot(time[10:],c_loss[10:],'b', label = 'Conditional loss')
plt.plot(time[10:],m_loss[10:],'r',label = 'Marginal loss')
#plt.plot(nt_time,nt_success,'k',label = 'GCSL_Norm_T_loss+')
plt.xlabel('TimeSteps')
plt.ylabel('Loss')
plt.legend()
plt.title('Losses -' + env )
plt.grid()
plt.show()
#plt.savefig('plots_new/'+env+'_loss/Success.jpg')
plt.close()

## Plot 2
plt.plot(time,avg_dist,'b', label = 'GCSL')
plt.plot(n_time,n_avg_dist,'r',label = 'GCSL_Norm_T')
#plt.plot(nt_time,nt_avg_dist,'k',label = 'GCSL_Norm_T_loss+')
plt.xlabel('TimeSteps')
plt.ylabel('Average Distance')
plt.legend()
plt.title('Average Distance -' + env)
plt.grid()
plt.show()
#plt.savefig('plots_new/'+ env +'_loss/avg_dist1.jpg')
plt.close()

plt.plot(time,loss,'b', label = 'GCSL')
plt.plot(n_time,n_loss,'r',label = 'GCSL_Norm_T')
#plt.plot(nt_time,nt_loss,'k',label = 'GCSL_Norm_T_loss+')
plt.xlabel('TimeSteps')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss -' + env )
plt.grid()
plt.show()
#plt.savefig('plots_new/'+env+'_loss/loss1.jpg')
plt.close()

##Plot3
'''''
plt.plot(time,dist_mean,'b', label = 'GCSL')
plt.fill_between(time,dist_mean - dist_std , dist_mean + dist_std,color = 'b' , alpha = 0.2)
plt.plot(n_time,n_dist_mean,'r',label = 'GCSL_Norm')
plt.fill_between(n_time,n_dist_mean - n_dist_std , n_dist_mean + n_dist_std,color = 'r' , alpha = 0.2)
plt.xlabel('TimeSteps')
plt.ylabel('Final Distance')
plt.legend()
plt.title('Final Distance -' + env)
plt.grid()
plt.show()
#plt.savefig('plots/' + env + '/3-final_dist.jpg')
plt.close()
'''''
print('K')
