import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
env = 'pointmass_rooms'

'''''
#door
df_gcsl_nt = pd.read_csv('/home/nsh1609/gcsl_bias/data/example/door/norm_offpb_s1_0/2022_01_24_22_55_01/progress.csv')
#df_gcsl_n = pd.read_csv('data/example/door/norm_sto/2022_01_11_20_20_38/progress.csv')
df_gcsl = pd.read_csv('/home/nsh1609/gcsl_bias/data/example/door/gcsl_offpb_s1_0/2022_01_24_22_55_01/progress.csv')
#df_gcsl_s = pd.read_csv('data/example/door/gcsl-sto/2022_01_11_20_20_38/progress.csv')
'''''
'''''
#pusher
df_gcsl_nt = pd.read_csv('example/pusher/norm_offpb_s5_0/2022_01_23_21_54_57/progress.csv')
#df_gcsl_n = pd.read_csv('data/example/pusher/norm_sto/2022_01_11_20_20_38/progress.csv')
df_gcsl = pd.read_csv('example/pusher/gcsl_offpb_s5_0/2022_01_23_21_54_57/progress.csv')
#df_gcsl_s = pd.read_csv('data/example/pusher/gcsl-sto/2022_01_11_20_20_37/progress.csv')
'''''
'''''
#lunar
df_gcsl_nt = pd.read_csv('data/example/lunar/norm_offpb_s3_0/2022_01_24_14_13_38/progress.csv')
#df_gcsl_n = pd.read_csv('data/example/lunar/gcsl_mt2_sto_1/2022_01_08_10_53_33/progress.csv')
df_gcsl = pd.read_csv('data/example/lunar/gcsl_offpb_s3_0/2022_01_24_14_13_38/progress.csv')
#df_gcsl_s = pd.read_csv('data/example/lunar/gcsl_o_1/2022_01_09_12_17_47/progress.csv')
'''''
'''''
#pointmass_rooms
df_gcsl_nt = pd.read_csv('example/pointmass_rooms/norm_offpb_s0_0/2022_01_24_12_33_35/progress.csv')
#df_gcsl_n = pd.read_csv('data/example/pointmass_rooms/gcsl_mt2_sto_1/2022_01_08_10_53_34/progress.csv')
df_gcsl = pd.read_csv('example/pointmass_rooms/gcsl_offpb_s0_0/2022_01_24_12_33_35/progress.csv')
#df_gcsl_s = pd.read_csv('data/example/pointmass_rooms/gcsl_o_1/2022_01_09_12_17_45/progress.csv')
'''''
#'''''
#pointmass_empty
df_gcsl_nt = pd.read_csv('data/example/pointmass_empty/norm_offpb_s2_0/2022_01_24_01_50_09/progress.csv')
#df_gcsl_n = pd.read_csv('data/example/pointmass_empty/gcsl_mt2_sto_1/2022_01_08_10_53_34/progress.csv')
df_gcsl = pd.read_csv('data/example/pointmass_empty/gcsl_offpb_s2_0/2022_01_24_01_50_09/progress.csv')
#df_gcsl_s = pd.read_csv('data/example/pointmass_empty/gcsl_o_1/2022_01_09_12_17_47/progress.csv')
#'''''

nt_time = df_gcsl_nt['timesteps'].values
nt_success = df_gcsl_nt['Eval success ratio'].values
nt_avg_dist = df_gcsl_nt['Eval avg final dist'].values
nt_loss = df_gcsl_nt['policy loss'].values

'''''
n_time = df_gcsl_n['timesteps'].values
n_success = df_gcsl_n['Eval success ratio'].values
n_avg_dist = df_gcsl_n['Eval avg final dist'].values
n_loss = df_gcsl_n['policy loss'].values
n_success_l = df_gcsl_n['Eval success ratio_left'].values
n_success_r = df_gcsl_n['Eval success ratio_right'].values
'''''

time = df_gcsl['timesteps'].values
success = df_gcsl['Eval success ratio'].values
avg_dist = df_gcsl['Eval avg final dist'].values
loss = df_gcsl['policy loss'].values
#dist_mean = df_gcsl['Eval final puck distance Mean'].values
#dist_std = df_gcsl['Eval final puck distance Std'].values
'''''
s_time = df_gcsl_s['timesteps'].values
s_success = df_gcsl_s['Eval success ratio'].values
s_avg_dist = df_gcsl_s['Eval avg final dist'].values
s_loss = df_gcsl_s['policy loss'].values
s_success_l = df_gcsl_s['Eval success ratio_left'].values
s_success_r = df_gcsl_s['Eval success ratio_right'].values
'''''
########################################################################################################################
#Seed_1

'''''
#door
df_gcsl_nt_1 = pd.read_csv('/home/nsh1609/gcsl_bias/data/example/door/norm_offpb_s1_1/2022_01_25_02_29_13/progress.csv')
#df_gcsl_n_1 = pd.read_csv('data/example/door/norm_sto/2022_01_12_00_12_12/progress.csv')
df_gcsl_1 = pd.read_csv('/home/nsh1609/gcsl_bias/data/example/door/gcsl_offpb_s1_1/2022_01_25_02_29_13/progress.csv')
#df_gcsl_s_1 = pd.read_csv('data/example/door/gcsl-sto/2022_01_12_00_12_12/progress.csv')
'''''
'''''
#pusher
df_gcsl_nt_1 = pd.read_csv('/home/nsh1609/gcsl_bias/example/pusher/norm_offpb_s5_1/2022_01_24_02_47_48/progress.csv')
#df_gcsl_n_1 = pd.read_csv('data/example/pusher/norm_sto/2022_01_12_00_12_11/progress.csv')
df_gcsl_1 = pd.read_csv('/home/nsh1609/gcsl_bias/example/pusher/gcsl_offpb_s5_1/2022_01_24_02_47_48/progress.csv')
#df_gcsl_s_1 = pd.read_csv('data/example/pusher/gcsl-sto/2022_01_12_00_12_11/progress.csv')
'''''
'''''
#lunar
df_gcsl_nt_1 = pd.read_csv('data/example/lunar/norm_offpb_s3_3/2022_01_24_16_15_03/progress.csv')
#df_gcsl_n_1 = pd.read_csv('data/example/lunar/gcsl_mt2_sto_1/2022_01_08_10_53_33/progress.csv')
df_gcsl_1 = pd.read_csv('data/example/lunar/gcsl_offpb_s3_3/2022_01_24_16_15_04/progress.csv')
#df_gcsl_s_1 = pd.read_csv('data/example/lunar/gcsl_o_1/2022_01_09_14_48_13/progress.csv')
'''''
'''''
#pointmass_rooms
df_gcsl_nt_1 = pd.read_csv('example/pointmass_rooms/norm_offpb_s0_5/2022_01_24_19_06_49/progress.csv')
#df_gcsl_n_1 = pd.read_csv('data/example/pointmass_rooms/gcsl_mt2_sto_1/2022_01_08_14_22_49/progress.csv')
df_gcsl_1 = pd.read_csv('example/pointmass_rooms/gcsl_offpb_s0_5/2022_01_24_19_06_49/progress.csv')
#df_gcsl_s_1 = pd.read_csv('data/example/pointmass_rooms/gcsl_o_1/2022_01_09_14_48_13/progress.csv')
'''''
#'''''
#pointmass_empty
df_gcsl_nt_1 = pd.read_csv('data/example/pointmass_empty/norm_offpb_s2_1/2022_01_24_02_24_27/progress.csv')
#df_gcsl_n_1 = pd.read_csv('data/example/pointmass_empty/gcsl_mt2_sto_1/2022_01_08_14_22_49/progress.csv')
df_gcsl_1 = pd.read_csv('data/example/pointmass_empty/gcsl_offpb_s2_1/2022_01_24_02_24_27/progress.csv')
#df_gcsl_s_1 = pd.read_csv('data/example/pointmass_empty/gcsl_o_1/2022_01_09_14_48_13/progress.csv')
#'''''

nt_time_1 = df_gcsl_nt_1['timesteps'].values
nt_success_1 = df_gcsl_nt_1['Eval success ratio'].values
nt_avg_dist_1 = df_gcsl_nt_1['Eval avg final dist'].values
nt_loss_1 = df_gcsl_nt_1['policy loss'].values

'''''
n_time_1 = df_gcsl_n_1['timesteps'].values
n_success_1 = df_gcsl_n_1['Eval success ratio'].values
n_avg_dist_1 = df_gcsl_n_1['Eval avg final dist'].values
n_loss_1 = df_gcsl_n_1['policy loss'].values
n_success_l_1 = df_gcsl_n_1['Eval success ratio_left'].values
n_success_r_1 = df_gcsl_n_1['Eval success ratio_right'].values
#n_dist_mean = df_gcsl_n['Eval final puck distance Mean'].values
#n_dist_std = df_gcsl_n['Eval final puck distance Std'].values
'''''

time_1 = df_gcsl_1['timesteps'].values
success_1 = df_gcsl_1['Eval success ratio'].values
avg_dist_1 = df_gcsl_1['Eval avg final dist'].values
loss_1 = df_gcsl_1['policy loss'].values
#dist_mean = df_gcsl['Eval final puck distance Mean'].values
#dist_std = df_gcsl['Eval final puck distance Std'].values
'''''
s_time_1 = df_gcsl_s_1['timesteps'].values
s_success_1 = df_gcsl_s_1['Eval success ratio'].values
s_avg_dist_1 = df_gcsl_s_1['Eval avg final dist'].values
s_loss_1 = df_gcsl_s_1['policy loss'].values
s_success_l_1 = df_gcsl_s_1['Eval success ratio_left'].values
s_success_r_1 = df_gcsl_s_1['Eval success ratio_right'].values
'''''
#######################################################################################################################
#Seed 2
'''''
#door
df_gcsl_nt_2 = pd.read_csv('/home/nsh1609/gcsl_bias/data/example/door/norm_offpb_s1_2/2022_01_25_09_26_36/progress.csv')
#df_gcsl_n_2 = pd.read_csv('data/example/door/norm_sto/2022_01_12_04_04_05/progress.csv')
df_gcsl_2 = pd.read_csv('/home/nsh1609/gcsl_bias/data/example/door/gcsl_offpb_s1_2/2022_01_25_09_26_36/progress.csv')
#df_gcsl_s_2 = pd.read_csv('data/example/door/gcsl-sto/2022_01_12_04_04_05/progress.csv')
'''''
'''''
#pusher
df_gcsl_nt_2 = pd.read_csv('/home/nsh1609/gcsl_bias/example/pusher/norm_offpb_s5_2/2022_01_24_07_37_22/progress.csv')
#df_gcsl_n_2 = pd.read_csv('data/example/pusher/norm_sto/2022_01_12_04_04_04/progress.csv')
df_gcsl_2 = pd.read_csv('/home/nsh1609/gcsl_bias/example/pusher/gcsl_offpb_s5_2/2022_01_24_07_37_22/progress.csv')
#df_gcsl_s_2 = pd.read_csv('data/example/pusher/gcsl-sto/2022_01_12_04_04_04/progress.csv')
'''''
'''''
#lunar
df_gcsl_nt_2 = pd.read_csv('data/example/lunar/norm_offpb_s3_6/2022_01_24_17_53_52/progress.csv')
#df_gcsl_n_2 = pd.read_csv('data/example/lunar/gcsl_mt2_sto_1/2022_01_08_17_51_31/progress.csv')
df_gcsl_2 = pd.read_csv('data/example/lunar/gcsl_offpb_s3_6/2022_01_24_17_53_52/progress.csv')
#df_gcsl_s_2 = pd.read_csv('data/example/lunar/gcsl_o_1/2022_01_09_17_21_24/progress.csv')
'''''
'''''
#pointmass_rooms
df_gcsl_nt_2 = pd.read_csv('example/pointmass_rooms/norm_offpb_s0_2/2022_01_24_14_33_33/progress.csv')
#df_gcsl_n_2 = pd.read_csv('data/example/pointmass_rooms/gcsl_mt2_sto_1/2022_01_08_17_51_31/progress.csv')
df_gcsl_2 = pd.read_csv('example/pointmass_rooms/gcsl_offpb_s0_2/2022_01_24_14_33_33/progress.csv')
#df_gcsl_s_2 = pd.read_csv('data/example/pointmass_rooms/gcsl_o_1/2022_01_09_17_21_24/progress.csv')
'''''
#'''''
#pointmass_empty
df_gcsl_nt_2 = pd.read_csv('data/example/pointmass_empty/norm_offpb_s2_2/2022_01_24_09_50_44/progress.csv')
#df_gcsl_n_2 = pd.read_csv('data/example/pointmass_empty/gcsl_mt2_sto_1/2022_01_08_17_51_32/progress.csv')
df_gcsl_2 = pd.read_csv('data/example/pointmass_empty/gcsl_offpb_s2_2/2022_01_24_09_50_44/progress.csv')
#df_gcsl_s_2 = pd.read_csv('data/example/pointmass_empty/gcsl_o_1/2022_01_09_17_21_24/progress.csv')
#'''''

nt_time_2 = df_gcsl_nt_2['timesteps'].values
nt_success_2 = df_gcsl_nt_2['Eval success ratio'].values
nt_avg_dist_2 = df_gcsl_nt_2['Eval avg final dist'].values
nt_loss_2 = df_gcsl_nt_2['policy loss'].values
'''''
n_time_2 = df_gcsl_n_2['timesteps'].values
n_success_2 = df_gcsl_n_2['Eval success ratio'].values
n_avg_dist_2 = df_gcsl_n_2['Eval avg final dist'].values
n_loss_2 = df_gcsl_n_2['policy loss'].values
n_success_l_2 = df_gcsl_n_2['Eval success ratio_left'].values
n_success_r_2 = df_gcsl_n_2['Eval success ratio_right'].values
#n_dist_mean = df_gcsl_n['Eval final puck distance Mean'].values
#n_dist_std = df_gcsl_n['Eval final puck distance Std'].values
'''''
time_2 = df_gcsl_2['timesteps'].values
success_2 = df_gcsl_2['Eval success ratio'].values
avg_dist_2 = df_gcsl_2['Eval avg final dist'].values
loss_2 = df_gcsl_2['policy loss'].values
#dist_mean = df_gcsl['Eval final puck distance Mean'].values
#dist_std = df_gcsl['Eval final puck distance Std'].values
'''''
s_time_2 = df_gcsl_s_2['timesteps'].values
s_success_2 = df_gcsl_s_2['Eval success ratio'].values
s_avg_dist_2 = df_gcsl_s_2['Eval avg final dist'].values
s_loss_2 = df_gcsl_s_2['policy loss'].values
s_success_l_2 = df_gcsl_s_2['Eval success ratio_left'].values
s_success_r_2 = df_gcsl_s_2['Eval success ratio_right'].values
'''''
#######################################################################################################################
#Average and Std deviations by seeds
stack_success = np.vstack((success,success_1,success_2))
#stack_s_success = np.vstack((s_success,s_success_1,s_success_2))
#stack_n_success = np.vstack((n_success,n_success_1,n_success_2))
stack_nt_success = np.vstack((nt_success,nt_success_1,nt_success_2))
'''''
stack_s_success_l = np.vstack((s_success_l,s_success_l_1,s_success_l_2))
stack_s_success_r = np.vstack((s_success_r,s_success_r_1,s_success_r_2))
stack_n_success_l = np.vstack((n_success_l,n_success_l_1,n_success_l_2))
stack_n_success_r = np.vstack((n_success_r,n_success_r_1,n_success_r_2))
'''
a_success = np.mean(stack_success,axis = 0)
#a_s_success = np.mean(stack_s_success,axis = 0)
#a_n_success = np.mean(stack_n_success,axis = 0)
a_nt_success = np.mean(stack_nt_success,axis = 0)
'''''
a_s_success_l = np.mean(stack_s_success_l,axis = 0)
a_s_success_r = np.mean(stack_s_success_r,axis = 0)
a_n_success_l = np.mean(stack_n_success_l,axis = 0)
a_n_success_r = np.mean(stack_n_success_r,axis = 0)
'''''
std_success = np.std(stack_success,axis = 0)
#std_s_success = np.std(stack_s_success,axis = 0)
#std_n_success = np.std(stack_n_success,axis = 0)
std_nt_success = np.std(stack_nt_success,axis = 0)
'''''
std_s_success_l = np.std(stack_s_success_l,axis = 0)
std_s_success_r = np.std(stack_s_success_r,axis = 0)
std_n_success_l = np.std(stack_n_success_l,axis = 0)
std_n_success_r = np.std(stack_n_success_r,axis = 0)
'''''
stack_avg_dist = np.vstack((avg_dist,avg_dist_1,avg_dist_2))
#stack_s_avg_dist = np.vstack((s_avg_dist,s_avg_dist_1,s_avg_dist_2))
#stack_n_avg_dist = np.vstack((n_avg_dist,n_avg_dist_1,n_avg_dist_2))
stack_nt_avg_dist = np.vstack((nt_avg_dist,nt_avg_dist_1,nt_avg_dist_2))

a_avg_dist = np.mean(stack_avg_dist,axis = 0)
#a_s_avg_dist = np.mean(stack_s_avg_dist,axis = 0)
#a_n_avg_dist = np.mean(stack_n_avg_dist,axis = 0)
a_nt_avg_dist = np.mean(stack_nt_avg_dist,axis = 0)

std_avg_dist = np.std(stack_avg_dist,axis = 0)
#std_s_avg_dist = np.std(stack_s_avg_dist,axis = 0)
#std_n_avg_dist = np.std(stack_n_avg_dist,axis = 0)
std_nt_avg_dist = np.std(stack_nt_avg_dist,axis = 0)

stack_loss = np.vstack((loss,loss_1,loss_2))
#stack_s_loss = np.vstack((s_loss,s_loss_1,s_loss_2))
#stack_n_loss = np.vstack((n_loss,n_loss_1,n_loss_2))
stack_nt_loss = np.vstack((nt_loss,nt_loss_1,nt_loss_2))

a_loss = np.mean(stack_loss,axis = 0)
#a_s_loss = np.mean(stack_s_loss,axis = 0)
#a_n_loss = np.mean(stack_n_loss,axis = 0)
a_nt_loss = np.mean(stack_nt_loss,axis = 0)

std_loss = np.std(stack_loss,axis = 0)
#std_s_loss = np.std(stack_s_loss,axis = 0)
#std_n_loss = np.std(stack_n_loss,axis = 0)
std_nt_loss = np.std(stack_nt_loss,axis = 0)

print('A-S',a_success[-1])
print('std-S',std_success[-1])
print('A-NS',a_nt_success[-1])
print('std-NS',std_nt_success[-1])

'''''
## Plot 1

#plt.plot(time,a_success,'m', label = 'GCSL_Det')
#plt.fill_between(time,a_success - std_success , a_success + std_success,color = 'm' , alpha = 0.2)
plt.plot(time,a_nt_success,'r',label = 'Normalized(Ours)')
plt.fill_between(time,a_nt_success - std_nt_success , a_nt_success + std_nt_success,color = 'r' , alpha = 0.2)
#plt.plot(time,a_nt_success,'g',label = 'Norm_Det')
#plt.fill_between(time,a_nt_success - std_nt_success , a_nt_success + std_nt_success,color = 'g' , alpha = 0.2)
plt.plot(time,a_success,'b',label = 'GCSL')
plt.fill_between(time,a_success - std_success , a_success + std_success,color = 'b' , alpha = 0.2)
plt.xlabel('TimeSteps')
plt.ylabel('Success Ratio')
plt.legend()
plt.title('Success Ratio -' + env )
plt.grid()
#plt.show()
plt.savefig('Plots_bias/'+env+'/success.jpg')
plt.close()

## Plot 2
#plt.plot(time,a_avg_dist,'m', label = 'GCSL_Det')
#plt.fill_between(time,a_avg_dist - std_avg_dist , a_avg_dist + std_avg_dist,color = 'm' , alpha = 0.2)
plt.plot(time,a_nt_avg_dist,'r',label = 'Normalized(Ours)')
plt.fill_between(time,a_nt_avg_dist - std_nt_avg_dist , a_nt_avg_dist + std_nt_avg_dist,color = 'r' , alpha = 0.2)
#plt.plot(time,a_nt_avg_dist,'g',label = 'Norm_Det')
#plt.fill_between(time,a_nt_avg_dist - std_nt_avg_dist , a_nt_avg_dist + std_nt_avg_dist,color = 'g' , alpha = 0.2)
plt.plot(time,a_avg_dist,'b',label = 'GCSL')
plt.fill_between(time,a_avg_dist - std_avg_dist , a_avg_dist + std_avg_dist,color = 'b' , alpha = 0.2)
plt.xlabel('TimeSteps')
plt.ylabel('Average Distance')
plt.legend()
plt.title('Average Distance -' + env )
plt.grid()
#plt.show()
plt.savefig('Plots_bias/'+env+'/avg_dist.jpg')
plt.close()

## Plot 3

#plt.plot(time,a_loss,'m', label = 'GCSL_Det')
#plt.fill_between(time,a_loss - std_loss , a_loss + std_loss,color = 'm' , alpha = 0.2)
plt.plot(time,a_nt_loss,'r',label = 'Normalized(Ours)')
plt.fill_between(time,a_nt_loss - std_nt_loss , a_nt_loss + std_nt_loss,color = 'r' , alpha = 0.2)
#plt.plot(time,a_nt_loss,'g',label = 'Norm_Det')
#plt.fill_between(time,a_nt_loss - std_nt_loss , a_nt_loss + std_nt_loss,color = 'g' , alpha = 0.2)
plt.plot(time,a_loss,'b',label = 'GCSL')
plt.fill_between(time,a_loss - std_loss , a_loss + std_loss,color = 'b' , alpha = 0.2)
plt.xlabel('TimeSteps')
plt.ylabel('Policy Loss')
plt.legend()
plt.title('Policy Loss -' + env )
plt.grid()
#plt.show()
plt.savefig('Plots_bias/'+env+'/loss.jpg')
plt.close()

#Plot 4

#plt.plot(time,a_success,'m', label = 'GCSL_Det')
#plt.fill_between(time,a_success - std_success , a_success + std_success,color = 'm' , alpha = 0.2)
plt.plot(time,a_n_success_l,'r',label = 'Norm_Sto-Left')
plt.fill_between(time,a_n_success_l - std_n_success_l , a_n_success_l + std_n_success_l,color = 'r' , alpha = 0.2)
#plt.plot(time,a_nt_success,'g',label = 'Norm_Det')
#plt.fill_between(time,a_nt_success - std_nt_success , a_nt_success + std_nt_success,color = 'g' , alpha = 0.2)
plt.plot(time,a_s_success_l,'b',label = 'GCSL_Sto-Left')
plt.fill_between(time,a_s_success_l - std_s_success_l , a_s_success_l + std_s_success_l,color = 'b' , alpha = 0.2)
plt.xlabel('TimeSteps')
plt.ylabel('Success Ratio')
plt.legend()
plt.title('Success Ratio - Left -' + env )
plt.grid()
plt.show()
#plt.savefig('plots_bias/'+env+'/success_left.jpg')
plt.close()

#Plot 4

#plt.plot(time,a_success,'m', label = 'GCSL_Det')
#plt.fill_between(time,a_success - std_success , a_success + std_success,color = 'm' , alpha = 0.2)
plt.plot(time,a_n_success_r,'r',label = 'Norm_Sto-Right')
plt.fill_between(time,a_n_success_r - std_n_success_r , a_n_success_r + std_n_success_r,color = 'r' , alpha = 0.2)
#plt.plot(time,a_nt_success,'g',label = 'Norm_Det')
#plt.fill_between(time,a_nt_success - std_nt_success , a_nt_success + std_nt_success,color = 'g' , alpha = 0.2)
plt.plot(time,a_s_success_r,'b',label = 'GCSL_Sto-Right')
plt.fill_between(time,a_s_success_r - std_s_success_r , a_s_success_r + std_s_success_r,color = 'b' , alpha = 0.2)
plt.xlabel('TimeSteps')
plt.ylabel('Success Ratio')
plt.legend()
plt.title('Success Ratio - Right -' + env )
plt.grid()
plt.show()
#plt.savefig('plots_bias/'+env+'/success_right.jpg')
plt.close()
'''''