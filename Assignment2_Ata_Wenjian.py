import math

import pyworld as pyworld
from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc

# read wav files
fs_m, music = wavfile.read('./Sounds/music.wav')
fs_female, female_s = wavfile.read('./Sounds/female.wav')
fs_male, male_s = wavfile.read('./Sounds/male.wav')

# # draw the plots w.r.t. time
# fig0 = plt.figure()
# plt.figure(figsize=(20, 12))
# # for the female speech
#
# #time-vector, convert samples to time by dividing with sampling frequency
# fig1 = plt.figure()
# time_f = np.linspace(0, len(female_s)/fs_female, num = len(female_s))
# plt.subplot(4, 3, 1)
# plt.ylabel("Amplitude")
# plt.xlabel("Seconds [s]")
# plt.title("Female speech")
# plt.plot(time_f, female_s)
#
# # zooming to 20ms segments
# # calculate the list length needed for 20ms segment
# time_dif = round(len(female_s)*0.02/(len(female_s)/fs_female))
# plt.subplot(4, 3, 2)
# plt.ylabel("Amplitude")
# plt.xlabel("Seconds [s]")
# plt.title("Female speech zoomed, unvoiced")
# # cut the unvoiced part
# plt.plot(time_f[len(female_s)-time_dif:len(female_s)], female_s[len(female_s)-time_dif:len(female_s)])
#
# plt.subplot(4, 3, 3)
# plt.ylabel("Amplitude")
# plt.xlabel("Seconds [s]")
# plt.title("Female speech zoomed, voiced")
# # cut the voiced part
# q_female = round(3*len(female_s)/10) # get the 0.3th index of the seq
# plt.plot(time_f[q_female-time_dif:q_female], female_s[q_female-time_dif:q_female])
#
# # for music signal
# time_m = np.linspace(0, len(music)/fs_m, num = len(music))
# plt.subplot(4, 3, 4)
# plt.ylabel("Amplitude")
# plt.xlabel("Seconds [s]")
# plt.title("Music signal")
# plt.plot(time_m, music)
#
# # zooming to 20ms segments
# # calculate the list length needed for 20ms segment
# time_dif = round(len(music)*0.02/(len(music)/fs_m))
# plt.subplot(4, 3, 5)
# plt.ylabel("Amplitude")
# plt.xlabel("Seconds [s]")
# plt.title("Music signal zoomed, unvoiced")
# # cut the unvoiced part
# plt.plot(time_m[len(music)-time_dif:len(music)], music[len(music)-time_dif:len(music)])
#
# plt.subplot(4, 3, 6)
# plt.ylabel("Amplitude")
# plt.xlabel("Seconds [s]")
# plt.title("Music signal zoomed, voiced")
# # cut the voiced part
# q_music = round(1*len(music)/10) # get the 0.1 index of the seq
# plt.plot(time_m[q_music-time_dif:q_music], music[q_music-time_dif:q_music])
#
# # for male signal
# time_m = np.linspace(0, len(male_s)/fs_male, num = len(male_s))
# plt.subplot(4, 3, 7)
# plt.ylabel("Amplitude")
# plt.xlabel("Seconds [s]")
# plt.title("Male speech")
# plt.plot(time_m, male_s)
#
# # zooming to 20ms segments
# # calculate the list length needed for 20ms segment
# time_dif = round(len(male_s)*0.02/(len(male_s)/fs_male))
# plt.subplot(4, 3, 8)
# plt.ylabel("Amplitude")
# plt.xlabel("Seconds [s]")
# plt.title("Male speech zoomed, unvoiced")
# # cut the unvoiced part
# plt.plot(time_m[len(male_s)-time_dif:len(male_s)], male_s[len(male_s)-time_dif:len(male_s)])
#
# plt.subplot(4, 3, 9)
# plt.ylabel("Amplitude")
# plt.xlabel("Seconds [s]")
# plt.title("Male speech zoomed, voiced")
#
# # cut the voiced part
# q_male = round(3*len(male_s)/4) # get the quarter index of the seq
# plt.plot(time_m[q_male-time_dif:q_male], male_s[q_male-time_dif:q_male])

# Spectrogram part
# fig2 = plt.figure()
# plt.ylabel("Frequency")
# plt.xlabel("Time")
# plt.title("Spectrogram, music")
# fftlen = round(0.03*fs_m)
# plt.specgram(music,cmap='coolwarm', NFFT=fftlen, Fs=fs_m, window=np.hanning(fftlen),scale='dB')
# plt.show()
# fig3 = plt.figure(figsize=(10,10))
# mfcc_music = mfcc(music,fs_m,numcep=26, winfunc=np.hanning)
# mfcc_music = mfcc_music.T
# rows_mean = np.mean(mfcc_music, axis=1)
# rows_std = np.std(mfcc_music, axis=1)
# mfcc_music_norm = (mfcc_music - rows_mean.reshape(-1, 1)) / rows_std.reshape(-1, 1)
# plt.matshow(mfcc_music_norm,cmap="coolwarm",fignum=1)
# plt.ylabel("Cepstral Coefficients")
# plt.xlabel("Frame")
# plt.title("Cepstrogram, music")
# plt.show()

# #male speech
# plt.subplot(4, 3, 11)
# plt.ylabel("Frequency")
# plt.xlabel("Time")
# plt.title("Spectrogram, male speech")
# fftlen = round(0.03*fs_male)
# plt.specgram(male_s,cmap='coolwarm',NFFT=fftlen, Fs=fs_male, window=np.hanning(fftlen),scale='dB')
#
# #music
# plt.subplot(2, 1, 2)
# plt.ylabel("Frequency")
# plt.xlabel("Time")
# plt.title("Spectrogram, music")
# fftlen = round(0.03*fs_m)
# plt.specgram(music,cmap='coolwarm', NFFT=fftlen, Fs=fs_m, window=np.hanning(fftlen),scale='dB')
#
# plt.show()
#
# # MFCC part -> Cepstrogram
# #female speech
# #fig2, axs = plt.subplots(2, 1, figsize=(20, 4))
# fig3 = plt.figure()
# mfcc_fe = mfcc(female_s,fs_female,numcep=26,winfunc=np.hanning)
# mfcc_fe = mfcc_fe.T
# # normalize -> every row
# rows_mean = np.mean(mfcc_fe, axis=1)
# rows_std = np.std(mfcc_fe, axis=1)
# mfcc_fe_norm = (mfcc_fe - rows_mean.reshape(-1, 1)) / rows_std.reshape(-1, 1)
# plt.matshow(mfcc_fe_norm,cmap="coolwarm",fignum=1)
# plt.ylabel("Cepstral Coefficients")
# plt.xlabel("Frame")
# plt.title("Cepstrogram, female speech")
# plt.show()
#
# #music
# fig4 = plt.figure(figsize=(8,6))
# mfcc_music = mfcc(music,fs_m,numcep=26, winfunc=np.hanning)
# mfcc_music = mfcc_music.T
# # normalize -> every row
# rows_mean = np.mean(mfcc_music, axis=1)
# rows_std = np.std(mfcc_music, axis=1)
# mfcc_music_norm = (mfcc_music - rows_mean.reshape(-1, 1)) / rows_std.reshape(-1, 1)
# #var = np.std(mfcc_music_norm[0])
# #plt.matshow(mfcc_music_norm,cmap="coolwarm",fignum=1)
# plt.matshow(mfcc_music_norm,cmap="coolwarm",fignum=1)
# plt.ylabel("Cepstral Coefficients")
# plt.xlabel("Frame")
# plt.title("Cepstrogram, music")
# plt.show()

# plt.figure(2)
# mfcc_ma = mfcc(male_s,fs_male, winlen=0.03,winfunc=np.hamming)
# plt.plot(mfcc_ma)



# # Plot the spectrogam of the same phrase of male and female speaker
# fig5 = plt.figure(figsize=(10,8))
# plt.subplot(1,2,1)
# plt.ylabel("Frequency")
# plt.xlabel("Time Frame")
# plt.title("Spectrogram, male speech")
# fftlen = round(0.03*fs_male)
# plt.specgram(male_s[len(male_s)-round(0.4*fs_male):len(male_s)],cmap='coolwarm',NFFT=fftlen, Fs=fs_male, window=np.hanning(fftlen),scale='dB')
#
# plt.subplot(1,2,2)
# plt.ylabel("Frequency")
# plt.xlabel("Time Frame")
# plt.title("Spectrogram, female speech")
# fftlen = round(0.03*fs_female)
# plt.specgram(female_s[len(female_s)-round(0.4*fs_female):len(female_s)],cmap='coolwarm',NFFT=fftlen, Fs=fs_female, window=np.hanning(fftlen),scale='dB')
# plt.show()
#
# # Plot the cepstrogram with the same phrase of male and female speaker
# fig6 = plt.figure(figsize=(10,8))
# mfcc_fe = mfcc(female_s[len(female_s)-round(0.4*fs_female):len(female_s)],fs_female,numcep=13,winfunc=np.hanning)
# mfcc_fe = mfcc_fe.T
# rows_mean = np.mean(mfcc_fe, axis=1)
# rows_std = np.std(mfcc_fe, axis=1)
# mfcc_fe_norm = (mfcc_fe - rows_mean.reshape(-1, 1)) / rows_std.reshape(-1, 1)
# plt.matshow(mfcc_fe_norm,cmap="coolwarm",fignum=1)
# plt.ylabel("Cepstral Coefficients")
# plt.xlabel("Time Frame")
# plt.title("Cepstrogram, female speech")
# plt.show()
#
# fig7 = plt.figure(figsize=(10,8))
# mfcc_ma = mfcc(male_s[len(male_s)-round(0.4*fs_male):len(male_s)],fs_male,numcep=13,winfunc=np.hanning)
# mfcc_ma = mfcc_ma.T
# rows_mean1 = np.mean(mfcc_ma, axis=1)
# rows_std1 = np.std(mfcc_ma, axis=1)
# mfcc_ma_norm = (mfcc_ma - rows_mean1.reshape(-1, 1)) / rows_std1.reshape(-1, 1)
# plt.matshow(mfcc_ma_norm,cmap="coolwarm",fignum=1)
# plt.ylabel("Cepstral Coefficients")
# plt.xlabel("Time Frame")
# plt.title("Cepstrogram, male speech")
# plt.show()

# Calculate the correlation matrices
fig1 = plt.figure()
plt.ylabel("Row")
plt.xlabel("Column")
plt.title("Correlation between spectral coefficient series, female speech")
f, t, Sxx = signal.spectrogram(female_s, fs_female)
Sxx = 10*np.log10(Sxx)
corr_matrix = np.corrcoef(Sxx)
plt.imshow(corr_matrix,cmap="gray")
plt.show()

fig2 = plt.figure()
mfcc_fe = mfcc(female_s,fs_female,numcep=13,winfunc=np.hanning)
mfcc_fe = mfcc_fe.T
rows_mean = np.mean(mfcc_fe, axis=1)
rows_std = np.std(mfcc_fe, axis=1)
mfcc_fe_norm = (mfcc_fe - rows_mean.reshape(-1, 1)) / rows_std.reshape(-1, 1)
corr_matrix = np.corrcoef(mfcc_fe_norm)
plt.imshow(corr_matrix,cmap="gray")
plt.ylabel("Row")
plt.xlabel("Column")
plt.title("Correlation between cepstral coefficient series, female speech")
plt.show()





