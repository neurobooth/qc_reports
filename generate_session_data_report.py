import sys
import os
import os.path as op
from os import walk
import argparse
import numpy as np
from h5io import read_hdf5
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

plt.rcParams['figure.dpi'] = 150

# --- Helper function for get task start and end times --- #
def _get_start_end_task_times(mdata):
    start_local_ts = []
    end_local_ts = []
    for ix, txt in enumerate(mdata['time_series']):
        if txt[0][:10] == 'Task_start':
            start_local_ts.append(mdata['time_stamps'][ix])
        elif txt[0][:8] == 'Task_end':
            end_local_ts.append(mdata['time_stamps'][ix])
    return start_local_ts, end_local_ts

# --- Helper function to get MOT target traces --- #
def _get_MOT_target_traces(mdata):
    target_dict = dict()
    for i in range(10):
        target_str = 'target_'+str(i)
        target_dict[target_str] = []

    # this ugly piece of code runs faster than beautiful code since the loop runs only once
    for ix, txt in enumerate(mdata['time_series']):
        if '!V TARGET_POS' in txt[0]:
            txt_split = txt[0].split(' ') 
            if txt_split[2]=='target_0':
                target_dict['target_0'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
            elif txt_split[2]=='target_1':
                target_dict['target_1'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
            elif txt_split[2]=='target_2':
                target_dict['target_2'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
            elif txt_split[2]=='target_3':
                target_dict['target_3'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
            elif txt_split[2]=='target_4':
                target_dict['target_4'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
            elif txt_split[2]=='target_5':
                target_dict['target_5'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
            elif txt_split[2]=='target_6':
                target_dict['target_6'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
            elif txt_split[2]=='target_7':
                target_dict['target_7'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
            elif txt_split[2]=='target_8':
                target_dict['target_8'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
            elif txt_split[2]=='target_9':
                target_dict['target_9'].append([mdata['time_stamps'][ix], int(txt_split[3][:-1]), int(txt_split[4])])
    
    return target_dict

# --- Helper function to get generic target traces --- #
def _get_target_traces(mdata):
    ts_ix = []
    x_coord = []
    y_coord = []
    for ix, txt in enumerate(mdata['time_series']):
        if '!V TARGET_POS' in txt[0]:
            ts_ix.append(ix)
            l = txt[0].split(' ')
            x_coord.append(int(l[3][:-1]))
            y_coord.append(int(l[4]))

    ctrl_ts = mdata['time_stamps'][ts_ix]
    return ctrl_ts, x_coord, y_coord

# --- Function to plot session data traces --- #
def plot_traces(data, task, session_id):
    
    # Column mapping of device data to be plotted - this should be immutable and never changes
    data_cols=dict()
    data_cols['Eyelink'] = [0,1,3,4]
    data_cols['Mouse'] = [0,1]
    data_cols['Mbient_BK'] = [1,2,3,4,5,6]
    data_cols['Mbient_LF'] = [1,2,3,4,5,6]
    data_cols['Mbient_LH'] = [1,2,3,4,5,6]
    data_cols['Mbient_RF'] = [1,2,3,4,5,6]
    data_cols['Mbient_RH'] = [1,2,3,4,5,6]
    data_cols['Mic'] = []
    
    fig, axs = plt.subplots(len(data.keys()), 1, sharex=True, figsize=[20,(5*(len(data.keys())+1))])
    if len(data.keys())==1:
        temp_list=[]
        temp_list.append(axs)
        axs = temp_list
    for ax in axs:
        ax.set_rasterized(True)
    
    for ix,ky in enumerate(data.keys()):
        
        # Legend labels for data to be plotted - this mutates for MOT and hence is being reinitialized for each task
        legend_dict=dict()
        legend_dict['Eyelink'] = ['R_gazeX', 'R_gazeY', 'L_gazeX', 'L_gazeY',]
        legend_dict['Mouse'] = ['mouse_x', 'mouse_y']
        legend_dict['Mbient_BK'] = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        legend_dict['Mbient_LF'] = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        legend_dict['Mbient_LH'] = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        legend_dict['Mbient_RF'] = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        legend_dict['Mbient_RH'] = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        legend_dict['Mic'] = ['amplitude']
        
        fs = np.nan

        if (ky != 'Mic') and ('Mbient' not in ky):
            # compute Sampling Rate (fs)
            if len(data[ky]['device_data']['time_stamps']) > 1:
                fs = 1/np.median(np.diff(data[ky]['device_data']['time_stamps']))
            
            # only plot data if lenght is greater than 1 - corrupted data is generally of length 1
            if len(data[ky]['device_data']['time_series']) > 1:
                # plot device data
                axs[ix].plot(data[ky]['device_data']['time_stamps'], data[ky]['device_data']['time_series'][:, data_cols[ky]])
            
            # plot target traces from marker data
            if task.split('_')[-1] != 'MOT':
                marker_ts, target_x, target_y = _get_target_traces(data[ky]['marker'])
                axs[ix].plot(marker_ts, target_x, drawstyle='steps-post', ls='--', alpha=0.8)
                axs[ix].plot(marker_ts, target_y, drawstyle='steps-post', ls='--', alpha=0.8)
                legend_dict[ky].append('target_x')
                legend_dict[ky].append('target_y')
            elif task.split('_')[-1] == 'MOT':
                targets_dict = _get_MOT_target_traces(data[ky]['marker'])
                for k in targets_dict:
                    axs[ix].plot(np.array(targets_dict[k])[:,0], np.array(targets_dict[k])[:,1], drawstyle='steps-post', ls='--', alpha=0.3)
                    legend_dict[ky].append(k+'_x')
                    axs[ix].plot(np.array(targets_dict[k])[:,0], np.array(targets_dict[k])[:,2], drawstyle='steps-post', ls='--', alpha=0.3)
                    legend_dict[ky].append(k+'_y')
                
                # getting MOT task start and end times
                task_start_ts, task_end_ts = _get_start_end_task_times(data[ky]['marker'])
                if task_start_ts and task_end_ts:
                    for start,end in zip(task_start_ts, task_end_ts):
                        axs[ix].axvline(x=start, color='green', ls='-.', lw=2)
                        axs[ix].axvline(x=end, color='red', ls='-.', lw=2)
            
            # add legends
            axs[ix].legend(legend_dict[ky], loc='center left', bbox_to_anchor=(1.02, 0.5))
            
            # generate title
            title = session_id+'___'+task+'___'+ky+'\n'+'Sampling Rate = '+"{:.2f}".format(fs)+'\n'+'Sample Size = '+str(len(data[ky]['device_data']['time_series']))
            # add title to plot
            axs[ix].set_title(title, loc='left', fontsize=14)
        elif ky == 'Mic':
            # read audio data
            audio_tstmp = data[ky]['device_data']['time_stamps']
            audio_ts = data[ky]['device_data']['time_series']
            chunk_len = audio_ts.shape[1]
            
            # restructure audio data
            audio_tstmp = np.insert(audio_tstmp, 0, audio_tstmp[0] - np.diff(audio_tstmp).mean())
            tstmps = []
            for i in range(audio_ts.shape[0]):
                tstmps.append(np.linspace(audio_tstmp[i], audio_tstmp[i+1], chunk_len))
            audio_tstmp_full = np.hstack(tstmps)
            audio_ts_full = np.hstack(audio_ts)
            
            # compute  Sampling Rate (fs)
            if len(data[ky]['device_data']['time_stamps']) > 1:
                fs = 1/np.median(np.diff(audio_tstmp_full))
            
            # only plot data if length is greater than 1
            if len(data[ky]['device_data']['time_series']) > 1:
                # plot audio data
                axs[ix].plot(audio_tstmp_full, audio_ts_full)
            # add legends
            axs[ix].legend(legend_dict[ky], loc='center left', bbox_to_anchor=(1.02, 0.5))
            # generate title
            title = session_id+'___'+task+'___'+ky+'\n'+'Sampling Rate = '+"{:.2f}".format(fs)+'\n'+'Sample Size = '+str(len(data[ky]['device_data']['time_series']))
            # add title to plot
            axs[ix].set_title(title, loc='left', fontsize=14)
            # getting vocalization task start and end times
            for vocal_task in ['ahh', 'gogogo', 'lalala', 'mememe', 'pataka']:
                if '_'.join(task.split('_')[1:]) == vocal_task:
                    task_start_ts, task_end_ts = _get_start_end_task_times(data[ky]['marker'])
                    if task_start_ts and task_end_ts:
                        for start,end in zip(task_start_ts, task_end_ts):
                            axs[ix].axvline(x=start, color='green', ls='-.', lw=2)
                            axs[ix].axvline(x=end, color='red', ls='-.', lw=2)
            ## END of Mic plotting
        elif 'Mbient' in ky:
            # compute  Sampling Rate (fs)
            if len(data[ky]['device_data']['time_stamps']) > 1:
                fs = 1/np.median(np.diff(data[ky]['device_data']['time_stamps']))
            
            # only plot data if length is greater than 1
            if len(data[ky]['device_data']['time_series']) > 1:
                # plot device data
                ax1 = axs[ix]
                ax1.set_prop_cycle(color=['red', 'green', 'blue'])
                ax1.plot(data[ky]['device_data']['time_stamps'], data[ky]['device_data']['time_series'][:, data_cols[ky][:3]])
                # plot gyroscope data on right y axis
                ax2 = ax1.twinx()
                ax2.set_prop_cycle(color=['cyan','magenta','orange'])
                ax2.plot(data[ky]['device_data']['time_stamps'], data[ky]['device_data']['time_series'][:, data_cols[ky][3:]])
                # add left y axis labels and legends
                ax1.set_ylabel('Acceleration', fontsize=12)
                ax1.legend(legend_dict[ky][:3], loc='lower left', bbox_to_anchor=(1.05, 0.5))
                # add right y axis labels and legends
                ax2.set_ylabel('Gyroscope', fontsize=12)
                ax2.legend(legend_dict[ky][3:], loc='upper left', bbox_to_anchor=(1.05, 0.5))
            
            # title goes primary axis - left y - referenced by subplot axis object
            title = session_id+'___'+task+'___'+ky+'\n'+'Sampling Rate = '+"{:.2f}".format(fs)+'\n'+'Sample Size = '+str(len(data[ky]['device_data']['time_series']))
            axs[ix].set_title(title, loc='left', fontsize=14)
            
            # getting movement task start and end times
            for movement_task in ['finger_nose', 'foot_tapping', 'sit_to_stand', 'altern_hand_mov', 'passage']:
                if '_'.join(task.split('_')[1:]) == movement_task:
                    task_start_ts, task_end_ts = _get_start_end_task_times(data[ky]['marker'])
                    if task_start_ts and task_end_ts:
                        for start,end in zip(task_start_ts, task_end_ts):
                            axs[ix].axvline(x=start, color='green', ls='-.', lw=2)
                            axs[ix].axvline(x=end, color='red', ls='-.', lw=2)
            
            ## END of Mbient plotting
    return fig, axs

#### Parsing command line arguments ####
prog_desc = """Generates patient data reports as pdf files
Requires sesssion id in the form of 'subject-id_session-date'
Searches default location for session data
use -h or --help flag for help
"""

arg_parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawDescriptionHelpFormatter,)
arg_parser.add_argument('--subj_id', required=True)
arg_parser.add_argument('--session_date', required=True)
arg_parser.add_argument('--data_dir', default='/autofs/nas/neurobooth/data')
arg_parser.add_argument('--processed_data_dir', default='/autofs/nas/neurobooth/processed_data')
arg_parser.add_argument('--save_dir', default=os.getcwd())
args = arg_parser.parse_args()
args = vars(args)
########################################

# master device list
devices=["Eyelink", "Mouse", "Mbient_BK", "Mbient_LF", "Mbient_LH", "Mbient_RF", "Mbient_RH", "Mic"]

# Reading parsed args
session_id = args['subj_id']+'_'+args['session_date']
session_path  = op.join(args['data_dir'], session_id)

print('\n  Session ID = ', session_id)
print('\n  Searching for session data in :', args['data_dir'])

if op.exists(session_path):
    fnames = []
    for (dirpath, dirnames, filenames) in walk(session_path):
        fnames.extend(filenames)
        break
else:
    sys.exit(f'\n  Could not find {session_id} data in {session_path}\n\n  EXITING\n\n')

performed_tasks = ['_'.join(task_session.split('_')[2:]) for task_session in list(dict.fromkeys([fname.split('_obs')[0] for fname in fnames if fname[-5:]=='.hdf5']))]
print(f'\n  Following tasks found:\n')
_ = [print(' ',pt) for pt in performed_tasks]
print()

# generating figures
figure_list=[]
for task in performed_tasks:
    
    # get files for task from all session files
    task_files=[]
    for file in fnames:
        if task in file and file[-5:]=='.hdf5':
            task_files.append(file)
    
    # select files that will be parsed
    files_to_parse=[]
    for device in devices:
        for file in task_files:
            if device in file:
                files_to_parse.append(file)
    
    data=dict()
    for device in devices:
        for file in files_to_parse:
            if device in file:
                data.update({device: read_hdf5(op.join(session_path, file))})
    print(f'  Plotting {task} data')
    fig, axs = plot_traces(data, task, session_id)
    figure_list.append(fig)

# plotting landmarks
print('\n  Searching for landmark data in :', args['processed_data_dir'])
landmark_path = op.join(args['processed_data_dir'], session_id)

plot_landmarks = False
if op.exists(landmark_path):
    plot_landmarks = True
    fnames = []
    for (dirpath, dirnames, filenames) in walk(landmark_path):
        fnames.extend(filenames)
        break
else:
    print(f'\n  Could not find {session_id} landmark data in {landmark_path}\n\n  Saving any plotted data\n\n')

if plot_landmarks:
    # might need to change this later to only read 'face_landmarks.hdf5' and read arm_landmarks separately
    landmark_tasks = ['_'.join(task_session.split('_')[2:]) for task_session in list(dict.fromkeys([fname.split('_obs')[0] for fname in fnames if fname[-5:]=='.hdf5']))]
    print(f'\n  Following landmarks found:\n')
    _ = [print(' ',lt) for lt in landmark_tasks]
    print()
    landmark_fig, axs = plt.subplots(len(fnames), 1, figsize=[20, 5*len(fnames)])
    if len(fnames)==1:
        temp_list=[]
        temp_list.append(axs)
        axs = temp_list
    for ax in axs:
        ax.set_rasterized(True)

    for ix, fname in enumerate(fnames):
        print(f'  Plotting {landmark_tasks[ix]} Face Landmarks')
        landmark_data = read_hdf5(os.path.join(landmark_path, fname))['device_data']
        axs[ix].plot(landmark_data['time_series'][:,::20,0])
        axs[ix].plot(landmark_data['time_series'][:,::20,1])
        title = session_id+'__'+landmark_tasks[ix]+'__'+'Face-Landmarks'
        axs[ix].set_title(title, loc='left', fontsize=14)
    figure_list.append(landmark_fig)

# saving pdf
print('\n  Saving pdf report...')
save_loc = op.join(args['save_dir'], session_id+"_session_data_report.pdf")
with matplotlib.backends.backend_pdf.PdfPages(save_loc) as pdf:
    for fig in figure_list:
        pdf.savefig(fig)
        plt.close(fig)
print('\n  Session data report saved at :', save_loc)
print()

