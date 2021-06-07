# import seaborn as sns
import pandas as pd
# from matplotlib import pyplot as plt
# from matplotlib.pyplot import specgram
from sklearn import preprocessing 
import numpy as np
import glob
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
from tqdm import tqdm
## this is data preparation file using minmax scaler since I read some where that the given BONN EEG data have different sensor type and thereby have different amplitude  range.. 

# getting data form the files
file1 = glob.glob("data/*/*.txt") 
file2=glob.glob("data/*/*.TXT")  
final_list=file2+file1


def generate_data(file_name,data1):

    #binning magnitude
    seq = list(data1['signal'])
    fft_real = np.around(np.fft.fft(seq).real,0)
    #print(fft_real)
    # splititng into bins
    bins_real= (300,200,100,50,40,30,20,10,0,-10,-20,-30,-40,-50,-100,-200,-300)
    digitized_real = np.digitize(fft_real, bins_real)
    digitized_real=np.bincount(digitized_real)
    #print(digitized_real)
    d_r =np.array(digitized_real).tolist()
    #binning stage for phase


    fft_imag = np.around(np.fft.fft(seq).imag)
    #print(fft_imag)
    bins_imag= (2000,1000,500,360,270,180,90,0,-90,-180,-270,-360,-500,-1000,-2000)
    digitized_imag = np.digitize(fft_imag, bins_imag)
    digitized_imag=np.bincount(digitized_imag)
    #print(digitized_imag)
    # let see if we can use the phase of signal as well later
    d_i=np.array(digitized_imag).tolist()
    
    final_list= d_r+d_i
   

  

    
    return final_list




#reading files

new_df=pd.DataFrame()
final_df=pd.DataFrame()
for i in tqdm(final_list):
    
    data=pd.read_csv(i)
    data.columns =['signal'] 
    scaler = preprocessing.MinMaxScaler((1,100))
    standard_df = scaler.fit_transform(data)
    data = pd.DataFrame(standard_df, index=data.index, columns=data.columns)
    data=data.round()
    start = 0
   
    for j in range(128):
        part_data = data[start:start+64]
        name=os.path.splitext(os.path.basename(i))[0]
        name=name[0]

        
        
        # do functions for part dataset
        generates= generate_data(i,part_data)
        start = start+32
       
        new_df = pd.DataFrame([generates])
        new_df['target']= name
        
        final_df = final_df.append(new_df,ignore_index=True)
    
    


final_df.to_hdf('data_with_phase.h5', key='df', mode='w')


    

       
        #break
    #break
     
    

# seq =list(data['signal'][:60])
# #print(seq)



# fft_fre = np.fft.fft(seq)
# print(fft_fre.real)


# fft_fre1 = np.fft.fftfreq(len(seq))


# plt.subplot(211)
# plt.plot(fft_fre1,fft_fre.real, label="Real part")
# plt.title("FFT in Frequency Domain")
# #plt.subplot(212)
# ##plt.plot(fft_fre1, fft_fre.imag,label="Imaginary part")

# # plt.xlabel("frequency (Hz)")

# # plt.show()


# # transform_fft=fft(seq,2)
# # print(transform_fft)

# # reducing noise

# # # check if have time to reduce noise
# # import noisereduce as nr
# # load data
# # rate, data = data
# # select section of data that is noise
# # noisy_part = data['signal'][:500]




# # # smooth_data = pd.Series(data['signal']).rolling(window=173).mean().plot(style='k')
# plt.show()
# plt.savefig()



# #print(data)
# # sns.set_theme(style="darkgrid")
# # #plt.specgram(data['signal'], NFFT=64, Fs=128, noverlap=32)

# # # Load an example dataset with long-form data


# # # # Plot the responses for different events and regions
# # sns.lineplot(data=data)
# # plt.show()