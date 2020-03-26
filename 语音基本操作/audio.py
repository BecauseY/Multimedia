import numpy as np
import wave
import scipy.signal as signal
import matplotlib.pyplot as plt
import lpc


def enframe(signal, nw, inc, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length=len(signal) #信号总长度
    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵[0-160],[40-200]...
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
    return frames*win,nf   #返回帧信号矩阵

def wavread(filename):

    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    f.close()
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    #画图
    time = np.arange(0, nframes) * (1.0 / framerate)
    plt.figure()
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Single channel wavedata")
    plt.plot(time, waveData)
    plt.show()
    waveData = np.reshape(waveData, [nframes, nchannels]).T

    return waveData


def calEnergy(Frame,nf,nw):
    '''
    计算每一帧的能量
    :param Frame: 要计算的帧数组
    :param nf: 数组行，多少个帧
    :param nw: 数组列，每个帧多少个采样点
    :return: 返回所有帧的能量，一个列表
    '''
    result=[]
    if nf==1:
        sum = 0
        for j in range(nw):
            sum = sum + Frame[j] * Frame[j]
        result.append(sum)
    else:
        for i in range(nf):
            sum=0
            for j in range(nw):
                sum=sum+Frame[i][j]*Frame[i][j]
            result.append(sum)
    return result

def calzhuo(Energy,nf):
    '''
    计算出浊音的索引
    :param Energy: 能量数组
    :param nf: 多少帧
    :return: 返回每一个浊音段对应的帧的索引
    '''
    result=[]
    cache=[]
    for i in range(nf):
        if Energy[i]>0.1:
            cache.append(i)
            if Energy[i+1]<0.1 and Energy[i+2]<0.1:
                result.append(cache)
                cache=[]
    return result


def calxiangguan(Frame,i,nw):
    '''
    计算一帧自相关
    :param Frame: 加窗后的帧
    :param i: 对应帧的索引
    :param nw: 该帧有多少
    :return: 返回该帧的基因频率
    '''
    sum=0
    result=[]
    result2=[]
    for k in range(0,100):
        for j in range(nw-k):
            sum=sum+Frame[i][j]*Frame[i][j+k]
        result.append(sum)
    #归一化
    for l in range(len(result)):
        result2.append(result[l]/result[0])
    return 8000/result2.index(max(result2[-80:]))




filename = '01234.wav'
data = wavread(filename)
nw = 160  #窗大小
inc = 40 #窗偏移
winfunc = signal.hamming(nw) #汉明窗
Frame,nf = enframe(data[0], nw, inc, winfunc) #分帧加窗
Energy=calEnergy(Frame,nf,nw) #计算每一帧的能量

#画出每一帧的能量图
x = np.arange(1,len(Energy)+1)
plt.figure()
plt.title(u'每一帧的能量')
plt.xlabel(u'帧数')
plt.ylabel(u'能量')
plt.plot(x,Energy)
plt.show()

#计算浊音对应的帧
zhuo=calzhuo(Energy,nf)
print(zhuo)

#计算浊音每一帧的基音频率
xiangguanindex=[]
for i in range(5):
    for j in zhuo[i]:
        xiangguanindexi=calxiangguan(Frame,j,nw)
        xiangguanindex.append(xiangguanindexi)
    xiangguanindex.append(0 )
#画图
x = np.arange(1,len(xiangguanindex)+1)
plt.figure()
plt.title(u'浊音每一帧的基音频率')
plt.ylabel(u'基音频率')
plt.xlabel(u'每一部分的帧')
plt.plot(x,xiangguanindex)
plt.show()


#求预测增益
ans=[]
for i in zhuo[0]:
    A,E,K=lpc.levinson_1d(Frame[i],10)
    pre= signal.lfilter(A,[1],Frame[i])
    err=Frame[i]-pre
    erreng=calEnergy(err,1,160)
    result=Energy[i]/erreng
    ans.append(result)
    if i==180 :
        plt.figure()
        plt.subplot(2,1,1)
        x = np.arange(1, len(Frame[180]) + 1)
        plt.plot(x, Frame[180])
        plt.title(u'原始180帧')
        plt.subplot(2, 1, 2)
        x = np.arange(1, len(pre) + 1)
        plt.plot(x, pre)
        plt.title(u'预测180帧')
#画图
x = np.arange(1,len(ans)+1)
plt.figure()
plt.title(u'数字0的预测增益')
plt.plot(x,ans)
plt.show()