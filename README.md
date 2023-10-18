# A2DFT
Audio-source separation in music using 2D Fourier Transform implemented in Python. \
Based on the publication - MUSIC/VOICE SEPARATION USING THE 2D FOURIER TRANSFORM - by - Prem Seetharaman, Fatemeh Pishdadian, Bryan Pardo

The program takes a *.wav file as input. \
Default filename is mixture.wav \
Output file - BG.wav and FG.wav \
\
\
Tunable parameters : \
\
'WindowLengthAlongRateAxis'  # can be varied from 15 through 100.\
Smaller values for the length result in leakage from the singing voice into the accompaniment, while larger values result in leakage from accompaniment into singing voice.
