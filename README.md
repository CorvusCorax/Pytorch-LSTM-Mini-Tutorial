
# PyTorch LSTM Example on Sin(t) Waveform Prediction
- Network & Solver definition
- Mini-Batch Training & Testing
- Inifnite time sequence inference
- Iterative Finetuning

Example equivalent to Caffe example https://github.com/CorvusCorax/Caffe-LSTM-Mini-Tutorial<br>
which is based on http://www.xiaoliangbai.com/2018/01/30/Caffe-LSTM-SinWaveform-Batch by Xiaoliang Bai.<br>


```python
import numpy as np
import math
import os
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import torch
```


```python
use_cuda = True

torch.manual_seed(0x1010)
np.random.seed(0x1010)
device = torch.device("cuda" if use_cuda else "cpu")

```


```python
# Use the sample generator from Tensorflow Sin(t) online
def generate_sample(f = 1.0, t0 = None, batch_size = 1, predict = 50, samples = 100):
    """
    Generates data samples.

    :param f: The frequency to use for all time series or None to randomize.
    :param t0: The time offset to use for all time series or None to randomize.
    :param batch_size: The number of time series to generate.
    :param predict: The number of future samples to generate.
    :param samples: The number of past (and current) samples to generate.
    :return: Tuple that contains the past times and values as well as the future times and values. In all outputs,
             each row represents one time series of the batch.
    """
    Fs = 100.0

    T = np.empty((batch_size, samples))
    Y = np.empty((batch_size, samples))
    FT = np.empty((batch_size, predict))
    FY = np.empty((batch_size, predict))

    _t0 = t0
    for i in range(batch_size):
        t = np.arange(0, samples + predict) / Fs
        if _t0 is None:
            t0 = np.random.rand() * 2 * np.pi
        else:
            t0 = _t0 + i/float(batch_size)

        freq = f
        if freq is None:
            freq = np.random.rand() * 3.5 + 0.5

        y = np.sin(2 * np.pi * freq * (t + t0))

        T[i, :] = t[0:samples]
        Y[i, :] = y[0:samples]

        FT[i, :] = t[samples:samples + predict]
        FY[i, :] = y[samples:samples + predict]

    return T, Y, FT, FY
```


```python
snapshot_prefix = 'lstm_demo_snapshot'
# Network Parameters
n_input = 1 # single input stream
n_steps = 100 # timesteps
n_hidden = 15 # hidden units in LSTM
n_outputs = 50 # predictions in future time
batch_size = 20 # batch of data
NO_INPUT_DATA = -2 # defined numeric value for network if no input data is available
# Training Parameters
n_train = 6000
n_display = 200
n_adamAlpha = 0.001
n_adamEpsilon = 0.02
```

## Training Network definition


```python
class LSTMDemoNetwork(torch.nn.Module):

    n_hidden = None
    hx = (None,None)
    
    # n_input is input timesteps, n_hidden is LSTM units, n_output is output timesteps
    def __init__(self, n_hidden):
        super(LSTMDemoNetwork, self).__init__()
        self.n_hidden=n_hidden
        
        self.lstm = torch.nn.LSTM(1,self.n_hidden,num_layers=1)
        self.ip1 = torch.nn.Linear(self.n_hidden,1)
        
        self.init_weights()
        
    def forward(self, x, hx=None):
        
        out, thx = self.lstm(x,hx=hx)
        self.hx = (thx[0].detach(),thx[1].detach())
        out = self.ip1(out)
        return out
    
    def init_weights(self):
        # Initialize forget gate bias to 1
        #
        # from https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
        #    .. math::
        #    \begin{array}{ll} \\
        #        i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
        #        f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
        #        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
        #        o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
        #        c_t = f_t * c_{(t-1)} + i_t * g_t \\
        #        h_t = o_t * \tanh(c_t) \\
        #    \end{array}
        #...
        #    Attributes:
        #    weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
        #      `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
        #       Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`
        #    weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
        #      `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`
        #    bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
        #      `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        #    bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
        #      `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
        #
        #
        # This is a common trick to speed up LSTM training.
        # Unfortunately Pytorch does not offer selective weight initialisation in the layer definition.
        
        #print(self.lstm.state_dict().keys())
        param = getattr(self.lstm, 'bias_hh_l0')
        param.data[self.n_hidden:2*self.n_hidden] = 1
        # IMPORTANT! For some reason Pytorch uses two bias terms in each term of the LSTM cell
        # this is not only unnecessary and increases the number of learnable parameters
        # it also makes training less stable when compared to caffe1
        # we force bias_ih to 0 and set learning rate to 0 as well
        param = getattr(self.lstm, 'bias_ih_l0')
        param.data[:] = 0
        param.requires_grad = False

        

```


```python
lstmModel = LSTMDemoNetwork(n_hidden).to(device)
```


```python
lossFunction = torch.nn.MSELoss().to(device)
```


```python
optimizer = torch.optim.Adam(lstmModel.parameters(),lr=n_adamAlpha,eps=n_adamEpsilon)
```

## Training


```python
# Train network
def train_single(model, lf, solver, niter, disp_step):
    train_loss = np.zeros(niter) # this is for plotting, later
    
    
    unknown = torch.tensor(np.ones((n_outputs,batch_size), dtype=np.float32) * NO_INPUT_DATA).to(device)
    for i in range(niter):
        _, batch_x, _, batch_y = generate_sample(f=None,
                                         t0=None,
                                         batch_size=batch_size,
                                         samples=n_steps,
                                         predict=n_outputs)
        # IMPORTANT: Caffe LSTM has time in first dimension and batch in second, so
        # batched training data needs to be transposed
        # (This is from caffe examples, but unless batch_first has been passed to LSTM initializer, pytorch does the same thing)
        batch_x = torch.tensor(batch_x.transpose(),dtype=torch.float32).to(device)
        batch_y = torch.tensor(batch_y.transpose(),dtype=torch.float32).to(device)
        
        combined_x = torch.cat((batch_x,unknown)).view([n_steps+n_outputs,batch_size,1])
        combined_y = torch.cat((batch_x,batch_y)).view([n_steps+n_outputs,batch_size,1])
        
        result = model(combined_x)
        
        loss = lf(combined_y,result)
        
        train_loss[i] = loss.detach().cpu().numpy()
        
        solver.zero_grad()
        loss.backward()
        solver.step()
        
        if i % disp_step == 0:
            if i==0:
                print("step ", i, ", loss = ", train_loss[i])
            else:
                print("step ", i, ", loss = ", train_loss[i], ", avg loss = ", np.average(train_loss[i-disp_step:i]))
    print("Finished training, iteration reached ", niter, " final loss = ", train_loss[niter-1],
        " final avg = ", np.average(train_loss[niter-disp_step:niter-1]))
    return train_loss

train_loss = train_single(lstmModel,lossFunction,optimizer,n_train,n_display)

#explicitly save snapshot if it has not been done yet
filename='%s_iter_%i.pt' % (snapshot_prefix,n_train)
print('saving snapshot to "%s"' % (filename))
torch.save(lstmModel.state_dict(),filename)

# plot loss value during training
plt.plot(np.arange(n_train), train_loss)
plt.show()
```

    step  0 , loss =  0.6100238561630249
    step  200 , loss =  0.46383559703826904 , avg loss =  0.5117676916718483
    step  400 , loss =  0.35188817977905273 , avg loss =  0.4015021505951881
    step  600 , loss =  0.27478936314582825 , avg loss =  0.29233527250587943
    step  800 , loss =  0.21736185252666473 , avg loss =  0.23865433923900128
    step  1000 , loss =  0.20815210044384003 , avg loss =  0.21169447161257268
    step  1200 , loss =  0.18245530128479004 , avg loss =  0.19248730778694154
    step  1400 , loss =  0.17951136827468872 , avg loss =  0.17836878329515457
    step  1600 , loss =  0.1621991991996765 , avg loss =  0.1723385511338711
    step  1800 , loss =  0.1653403639793396 , avg loss =  0.16596302539110183
    step  2000 , loss =  0.1583380550146103 , avg loss =  0.162551117092371
    step  2200 , loss =  0.1612466424703598 , avg loss =  0.1598108634352684
    step  2400 , loss =  0.16245587170124054 , avg loss =  0.15708779029548167
    step  2600 , loss =  0.15433980524539948 , avg loss =  0.1543604374676943
    step  2800 , loss =  0.14505650103092194 , avg loss =  0.1518767724186182
    step  3000 , loss =  0.13852189481258392 , avg loss =  0.1487720350176096
    step  3200 , loss =  0.12218562513589859 , avg loss =  0.14528122432529927
    step  3400 , loss =  0.14171256124973297 , avg loss =  0.14186786215752364
    step  3600 , loss =  0.1320994347333908 , avg loss =  0.13702102996408938
    step  3800 , loss =  0.13346511125564575 , avg loss =  0.14042050909250975
    step  4000 , loss =  0.1315401941537857 , avg loss =  0.1420031564310193
    step  4200 , loss =  0.13872137665748596 , avg loss =  0.1294499559700489
    step  4400 , loss =  0.10143540054559708 , avg loss =  0.12134559355676174
    step  4600 , loss =  0.16664937138557434 , avg loss =  0.12554681070148946
    step  4800 , loss =  0.12061886489391327 , avg loss =  0.12734253853559493
    step  5000 , loss =  0.08962979912757874 , avg loss =  0.11184824105352163
    step  5200 , loss =  0.0747600719332695 , avg loss =  0.1083226903155446
    step  5400 , loss =  0.1375703066587448 , avg loss =  0.10364189278334379
    step  5600 , loss =  0.08898063004016876 , avg loss =  0.10170481231063605
    step  5800 , loss =  0.08319579809904099 , avg loss =  0.09477913180366158
    Finished training, iteration reached  6000  final loss =  0.11366216838359833  final avg =  0.09371602690624233
    saving snapshot to "lstm_demo_snapshot_iter_6000.pt"



![png](output_11_1.png)


## Testing


```python
# Test the prediction with trained (unrolled) net
# we can change the batch size on the network at runtime, but not the number of timesteps (depth of unrolling)
def test_net(net,n_tests):
    batch_size = 1

    unknown = torch.tensor(np.ones((n_outputs,batch_size), dtype=np.float32) * NO_INPUT_DATA).to(device)
    for i in range(1, n_tests + 1):
        plt.subplot(n_tests, 1, i)
        t, y, next_t, expected_y = generate_sample(f=i+0.1337, t0=None, samples=n_steps, predict=n_outputs)
        test_input = torch.tensor(y.transpose(),dtype=torch.float32).to(device)
        combined_x = torch.cat((test_input,unknown)).view([n_steps+n_outputs,batch_size,1])
        expected_y = expected_y.reshape(n_outputs)
        
        prediction = net(combined_x).detach().cpu().numpy()
        
        # remove the batch size dimensions
        t = t.squeeze()
        y = y.squeeze()
        next_t = next_t.squeeze()
        t2 = np.append(t,next_t)
        prediction = prediction.squeeze()
        
        plt.plot(t, y, color='black')
        plt.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle=":")
        plt.plot(t2, prediction, color='red')
        plt.ylim([-1, 1])
        plt.xlabel('time [t]')
        plt.ylabel('signal')
    plt.show()
test_net(lstmModel,3)
```


![png](output_13_0.png)


## Inference / Validation


```python
# the single step network can process infinite time series in a loop,
# as such we can increate n_outputs safely to have a glance at long term behaviour

def test_net_iterative(net,n_tests,n_outputs):
    for i in range(1, n_tests + 1):
        plt.subplot(n_tests, 1, i)
        t, y, next_t, expected_y = generate_sample(f=i+0.1337, t0=None, samples=n_steps, predict=n_outputs)
        expected_y = expected_y.reshape(n_outputs)

        prediction = []
        iA = np.zeros((1,1,1),dtype=np.float32)
        hx = None
        for T in range(n_steps):
            iA[0,0,0] = y[0,T].copy()
            x = torch.tensor(iA).to(device)
            yp = net(x,hx=hx).detach().cpu().numpy()
            hx = net.hx
            prediction.append(yp[0,0,0])
        
        for T in range(n_outputs):
            # in this case we have to manually indicate to the network
            # that there is no more input data at the current time step
            iA[0,0,0] = NO_INPUT_DATA
            x = torch.tensor(iA).to(device)
            yp = net(x,hx=hx).detach().cpu().numpy()
            hx = net.hx
            prediction.append(yp[0,0,0])
        
        # remove the batch size dimensions
        t = t.squeeze()
        y = y.squeeze()
        next_t = next_t.squeeze()
        t2 = np.append(t,next_t)
        prediction = np.array(prediction)
        
        plt.plot(t, y, color='black')
        plt.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle=":")
        plt.plot(t2, prediction, color='red')
        plt.ylim([-1, 1])
        plt.xlabel('time [t]')
        plt.ylabel('signal')
    plt.show()

test_net_iterative(lstmModel,3,600)
```


![png](output_15_0.png)


### Observation:
The network drifts towards a generic sine wave at constant frequency when left running for longer than the training sample size.
What happens if we train with a longer training window?


```python
n_outputs = 200
n_train = 20000
snapshot_prefix = 'lstm_demo2_snapshot'

lstmModel = LSTMDemoNetwork(n_hidden).to(device)
optimizer = torch.optim.Adam(lstmModel.parameters(),lr=n_adamAlpha,eps=n_adamEpsilon)
train_loss = train_single(lstmModel,lossFunction,optimizer,n_train,800)

#explicitly save snapshot if it has not been done yet
filename='%s_iter_%i.pt' % (snapshot_prefix,n_train)
print('saving snapshot to "%s"' % (filename))
torch.save(lstmModel.state_dict(),filename)


plt.plot(np.arange(n_train), train_loss)
plt.show()
```

    step  0 , loss =  0.5993098020553589
    step  800 , loss =  0.36738312244415283 , avg loss =  0.4462436730414629
    step  1600 , loss =  0.34903058409690857 , avg loss =  0.3582467787340283
    step  2400 , loss =  0.3405740559101105 , avg loss =  0.33886114452034233
    step  3200 , loss =  0.32634323835372925 , avg loss =  0.33354020312428473
    step  4000 , loss =  0.3298587501049042 , avg loss =  0.33072057999670507
    step  4800 , loss =  0.3283030688762665 , avg loss =  0.3287493133544922
    step  5600 , loss =  0.3225407004356384 , avg loss =  0.3270166611671448
    step  6400 , loss =  0.3237491548061371 , avg loss =  0.32480167396366594
    step  7200 , loss =  0.3187989890575409 , avg loss =  0.3212342968955636
    step  8000 , loss =  0.3114610016345978 , avg loss =  0.31326253194361925
    step  8800 , loss =  0.30659493803977966 , avg loss =  0.3122813962772489
    step  9600 , loss =  0.31901195645332336 , avg loss =  0.3151634018868208
    step  10400 , loss =  0.29218778014183044 , avg loss =  0.30723592575639486
    step  11200 , loss =  0.2900225818157196 , avg loss =  0.3011150689050555
    step  12000 , loss =  0.28400006890296936 , avg loss =  0.29929715652018785
    step  12800 , loss =  0.3201054334640503 , avg loss =  0.3114289540797472
    step  13600 , loss =  0.28136447072029114 , avg loss =  0.3051535439863801
    step  14400 , loss =  0.29169967770576477 , avg loss =  0.29688007034361363
    step  15200 , loss =  0.32572659850120544 , avg loss =  0.32527332320809366
    step  16000 , loss =  0.32227620482444763 , avg loss =  0.32703731760382654
    step  16800 , loss =  0.3101913034915924 , avg loss =  0.3207817366719246
    step  17600 , loss =  0.30284586548805237 , avg loss =  0.3142517460882664
    step  18400 , loss =  0.3241356611251831 , avg loss =  0.31600836109369995
    step  19200 , loss =  0.3086712956428528 , avg loss =  0.3166497207805514
    Finished training, iteration reached  20000  final loss =  0.3029806613922119  final avg =  0.308760990748865
    saving snapshot to "lstm_demo2_snapshot_iter_20000.pt"



![png](output_17_1.png)



```python
test_net_iterative(lstmModel,3,600)
```


![png](output_18_0.png)


### Observation:
With the longer unrolling window, the training converges much slower. The loss accumulated at the end of the window needs to backpropagate many steps until it reaches a timestep in which there is still a useful memory in the LSTM layer. This makes training potentially unstable.

A better option is to attempt iterative fine tuning with slowly increasing time windows. As a bonus, this allows doing most of the training with shorter windows, which means smaller networks and faster computation.

## Iterative Finetuning


```python
n_display = 400
n_outputs = 50
n_train = 10000

snapshot_prefix = 'lstm_demo3_0_snapshot'

print("initial training ", n_outputs," ouput timesteps for ",n_train," training cycles")

lstmModel = LSTMDemoNetwork(n_hidden).to(device)
optimizer = torch.optim.Adam(lstmModel.parameters(),lr=n_adamAlpha,eps=n_adamEpsilon)

train_loss = train_single(lstmModel,lossFunction,optimizer,n_train,n_display)

#explicitly save snapshot if it has not been done yet
filename='%s_iter_%i.pt' % (snapshot_prefix,n_train)
print('saving snapshot to "%s"' % (filename))
torch.save(lstmModel.state_dict(),filename)
net0=filename

plt.plot(np.arange(n_train), train_loss)
plt.show()

test_net_iterative(lstmModel,3,600)
```

    initial training  50  ouput timesteps for  10000  training cycles
    step  0 , loss =  0.5674645304679871
    step  400 , loss =  0.30983877182006836 , avg loss =  0.42093255557119846
    step  800 , loss =  0.24238353967666626 , avg loss =  0.25268871303647755
    step  1200 , loss =  0.18918123841285706 , avg loss =  0.20987924225628377
    step  1600 , loss =  0.18205870687961578 , avg loss =  0.18941344663500786
    step  2000 , loss =  0.16308528184890747 , avg loss =  0.1753477780148387
    step  2400 , loss =  0.1559123545885086 , avg loss =  0.16605412032455205
    step  2800 , loss =  0.14058952033519745 , avg loss =  0.15445086445659398
    step  3200 , loss =  0.13714367151260376 , avg loss =  0.14846248131245374
    step  3600 , loss =  0.1421949714422226 , avg loss =  0.14334236338734627
    step  4000 , loss =  0.13915038108825684 , avg loss =  0.1399128881841898
    step  4400 , loss =  0.13092833757400513 , avg loss =  0.13678157048299908
    step  4800 , loss =  0.11198166757822037 , avg loss =  0.13241701954975724
    step  5200 , loss =  0.11275465041399002 , avg loss =  0.12056515583768487
    step  5600 , loss =  0.10991053283214569 , avg loss =  0.1133300599269569
    step  6000 , loss =  0.1106618270277977 , avg loss =  0.11166221337392926
    step  6400 , loss =  0.10361015051603317 , avg loss =  0.10439997596666217
    step  6800 , loss =  0.08229666948318481 , avg loss =  0.09271162228658796
    step  7200 , loss =  0.07472217082977295 , avg loss =  0.07909428033977747
    step  7600 , loss =  0.06766573339700699 , avg loss =  0.0681962291803211
    step  8000 , loss =  0.0323370136320591 , avg loss =  0.05369881788268685
    step  8400 , loss =  0.027782050892710686 , avg loss =  0.035783933475613594
    step  8800 , loss =  0.022237971425056458 , avg loss =  0.02789714374113828
    step  9200 , loss =  0.027625788003206253 , avg loss =  0.02477482489310205
    step  9600 , loss =  0.025524307042360306 , avg loss =  0.01947603355627507
    Finished training, iteration reached  10000  final loss =  0.018504885956645012  final avg =  0.01802341028917254
    saving snapshot to "lstm_demo3_0_snapshot_iter_10000.pt"



![png](output_21_1.png)



![png](output_21_2.png)



```python
n_outputs = 100
n_train = 6000

snapshot_prefix = 'lstm_demo3_1_snapshot'

print("initial training ", n_outputs," ouput timesteps for ",n_train," training cycles")

lstmModel = LSTMDemoNetwork(n_hidden).to(device)
# load weights:
lstmModel.load_state_dict(torch.load(net0))
optimizer = torch.optim.Adam(lstmModel.parameters(),lr=n_adamAlpha,eps=n_adamEpsilon)

train_loss = train_single(lstmModel,lossFunction,optimizer,n_train,n_display)

#explicitly save snapshot if it has not been done yet
filename='%s_iter_%i.pt' % (snapshot_prefix,n_train)
print('saving snapshot to "%s"' % (filename))
torch.save(lstmModel.state_dict(),filename)
net1=filename

plt.plot(np.arange(n_train), train_loss)
plt.show()

test_net_iterative(lstmModel,3,600)
```

    initial training  100  ouput timesteps for  6000  training cycles
    step  0 , loss =  0.07179751992225647
    step  400 , loss =  0.04386515915393829 , avg loss =  0.061182753699831664
    step  800 , loss =  0.09873643517494202 , avg loss =  0.04699650762602687
    step  1200 , loss =  0.023680172860622406 , avg loss =  0.04237248070538044
    step  1600 , loss =  0.04869146645069122 , avg loss =  0.037694152360782024
    step  2000 , loss =  0.019881540909409523 , avg loss =  0.030602476422209292
    step  2400 , loss =  0.024556143209338188 , avg loss =  0.030611779880709945
    step  2800 , loss =  0.04189608246088028 , avg loss =  0.030109703775960953
    step  3200 , loss =  0.01703086495399475 , avg loss =  0.030579789385665208
    step  3600 , loss =  0.015673721209168434 , avg loss =  0.027010483865160494
    step  4000 , loss =  0.05981170013546944 , avg loss =  0.025126901678740977
    step  4400 , loss =  0.014695246703922749 , avg loss =  0.022143100197426976
    step  4800 , loss =  0.013369301334023476 , avg loss =  0.02178293542470783
    step  5200 , loss =  0.018563488498330116 , avg loss =  0.024162344792857768
    step  5600 , loss =  0.022386537864804268 , avg loss =  0.0228674180386588
    Finished training, iteration reached  6000  final loss =  0.021470926702022552  final avg =  0.019623865252057265
    saving snapshot to "lstm_demo3_1_snapshot_iter_6000.pt"



![png](output_22_1.png)



![png](output_22_2.png)



```python
n_outputs = 200
n_train = 8000

snapshot_prefix = 'lstm_demo3_2_snapshot'

print("initial training ", n_outputs," ouput timesteps for ",n_train," training cycles")

lstmModel = LSTMDemoNetwork(n_hidden).to(device)
# load weights:
lstmModel.load_state_dict(torch.load(net1))
optimizer = torch.optim.Adam(lstmModel.parameters(),lr=n_adamAlpha,eps=n_adamEpsilon)

train_loss = train_single(lstmModel,lossFunction,optimizer,n_train,n_display)

#explicitly save snapshot if it has not been done yet
filename='%s_iter_%i.pt' % (snapshot_prefix,n_train)
print('saving snapshot to "%s"' % (filename))
torch.save(lstmModel.state_dict(),filename)
net2=filename

plt.plot(np.arange(n_train), train_loss)
plt.show()

test_net_iterative(lstmModel,3,600)
```

    initial training  200  ouput timesteps for  8000  training cycles
    step  0 , loss =  0.07823554426431656
    step  400 , loss =  0.3082226812839508 , avg loss =  0.31507726760581134
    step  800 , loss =  0.28302016854286194 , avg loss =  0.28704889595508576
    step  1200 , loss =  0.23874340951442719 , avg loss =  0.2718718259036541
    step  1600 , loss =  0.22829364240169525 , avg loss =  0.2529639072716236
    step  2000 , loss =  0.18757495284080505 , avg loss =  0.24449665740132331
    step  2400 , loss =  0.3054700791835785 , avg loss =  0.24931358870118855
    step  2800 , loss =  0.2238008826971054 , avg loss =  0.2374013230577111
    step  3200 , loss =  0.11111718416213989 , avg loss =  0.19704528357833623
    step  3600 , loss =  0.09020867943763733 , avg loss =  0.15532408263534309
    step  4000 , loss =  0.15085208415985107 , avg loss =  0.10852161114104092
    step  4400 , loss =  0.06994427740573883 , avg loss =  0.09443443451076745
    step  4800 , loss =  0.0706186443567276 , avg loss =  0.0783108841907233
    step  5200 , loss =  0.04296185076236725 , avg loss =  0.06777180855162442
    step  5600 , loss =  0.04341905564069748 , avg loss =  0.04988356251269579
    step  6000 , loss =  0.060596633702516556 , avg loss =  0.05368827513884753
    step  6400 , loss =  0.031181370839476585 , avg loss =  0.05029259566683322
    step  6800 , loss =  0.06263475120067596 , avg loss =  0.052052143262699246
    step  7200 , loss =  0.034290213137865067 , avg loss =  0.044606765871867536
    step  7600 , loss =  0.043462689965963364 , avg loss =  0.046963601331226526
    Finished training, iteration reached  8000  final loss =  0.016892900690436363  final avg =  0.04482044198113799
    saving snapshot to "lstm_demo3_2_snapshot_iter_8000.pt"



![png](output_23_1.png)



![png](output_23_2.png)



```python
n_outputs = 400
n_train = 20000

snapshot_prefix = 'lstm_demo3_2_snapshot'

print("initial training ", n_outputs," ouput timesteps for ",n_train," training cycles")

lstmModel = LSTMDemoNetwork(n_hidden).to(device)
# load weights:
lstmModel.load_state_dict(torch.load(net2))
optimizer = torch.optim.Adam(lstmModel.parameters(),lr=n_adamAlpha/2,eps=n_adamEpsilon*2)

train_loss = train_single(lstmModel,lossFunction,optimizer,n_train,n_display)

#explicitly save snapshot if it has not been done yet
filename='%s_iter_%i.pt' % (snapshot_prefix,n_train)
print('saving snapshot to "%s"' % (filename))
torch.save(lstmModel.state_dict(),filename)
net3=filename

plt.plot(np.arange(n_train), train_loss)
plt.show()

test_net_iterative(lstmModel,3,600)
```

    initial training  400  ouput timesteps for  20000  training cycles
    step  0 , loss =  0.2417384684085846
    step  400 , loss =  0.3802364766597748 , avg loss =  0.3961190339922905
    step  800 , loss =  0.36070266366004944 , avg loss =  0.34978813380002977
    step  1200 , loss =  0.31221869587898254 , avg loss =  0.3209123172610998
    step  1600 , loss =  0.31287479400634766 , avg loss =  0.3358392792195082
    step  2000 , loss =  0.37020471692085266 , avg loss =  0.3133617028594017
    step  2400 , loss =  0.318511426448822 , avg loss =  0.3158454853668809
    step  2800 , loss =  0.3617900311946869 , avg loss =  0.3181963883712888
    step  3200 , loss =  0.23838287591934204 , avg loss =  0.3042260016128421
    step  3600 , loss =  0.24795377254486084 , avg loss =  0.29421926647424695
    step  4000 , loss =  0.21419012546539307 , avg loss =  0.2833383076637983
    step  4400 , loss =  0.37373048067092896 , avg loss =  0.2854553144797683
    step  4800 , loss =  0.3574375510215759 , avg loss =  0.2726440889015794
    step  5200 , loss =  0.24906352162361145 , avg loss =  0.24820176795125007
    step  5600 , loss =  0.3088309168815613 , avg loss =  0.28066089980304243
    step  6000 , loss =  0.3630617558956146 , avg loss =  0.2487983927503228
    step  6400 , loss =  0.15489281713962555 , avg loss =  0.2558498025685549
    step  6800 , loss =  0.1724589318037033 , avg loss =  0.2361430866457522
    step  7200 , loss =  0.17813941836357117 , avg loss =  0.22636637752875685
    step  7600 , loss =  0.22419112920761108 , avg loss =  0.19472642172127963
    step  8000 , loss =  0.12985140085220337 , avg loss =  0.19410479166544975
    step  8400 , loss =  0.1421784907579422 , avg loss =  0.17879845025017857
    step  8800 , loss =  0.05857166647911072 , avg loss =  0.11427636014297604
    step  9200 , loss =  0.07518190890550613 , avg loss =  0.1337810768187046
    step  9600 , loss =  0.12090877443552017 , avg loss =  0.123783427067101
    step  10000 , loss =  0.07604173570871353 , avg loss =  0.10479012469761073
    step  10400 , loss =  0.055052317678928375 , avg loss =  0.09160819497890771
    step  10800 , loss =  0.24722011387348175 , avg loss =  0.10011224077548832
    step  11200 , loss =  0.1114102303981781 , avg loss =  0.08324208219535649
    step  11600 , loss =  0.14068636298179626 , avg loss =  0.0915181951224804
    step  12000 , loss =  0.06901068985462189 , avg loss =  0.08120087944436818
    step  12400 , loss =  0.2151074856519699 , avg loss =  0.10351389294490218
    step  12800 , loss =  0.04228517785668373 , avg loss =  0.09186296372208744
    step  13200 , loss =  0.06884298473596573 , avg loss =  0.07728165757376701
    step  13600 , loss =  0.07363653182983398 , avg loss =  0.07830392704810947
    step  14000 , loss =  0.07472634315490723 , avg loss =  0.08215399933978915
    step  14400 , loss =  0.1202554926276207 , avg loss =  0.08858864304609597
    step  14800 , loss =  0.04567205160856247 , avg loss =  0.09998644658830017
    step  15200 , loss =  0.07285066694021225 , avg loss =  0.07876378685235977
    step  15600 , loss =  0.048514705151319504 , avg loss =  0.07967117227613926
    step  16000 , loss =  0.1159677803516388 , avg loss =  0.08072371372487396
    step  16400 , loss =  0.16969966888427734 , avg loss =  0.07685292581096291
    step  16800 , loss =  0.03979143127799034 , avg loss =  0.07849125532899053
    step  17200 , loss =  0.051311057060956955 , avg loss =  0.08302943719550968
    step  17600 , loss =  0.09293260425329208 , avg loss =  0.0804610605398193
    step  18000 , loss =  0.06861778348684311 , avg loss =  0.08652417317964137
    step  18400 , loss =  0.10969476401805878 , avg loss =  0.0762527915276587
    step  18800 , loss =  0.08178454637527466 , avg loss =  0.06858136689290405
    step  19200 , loss =  0.18140973150730133 , avg loss =  0.06844571948517114
    step  19600 , loss =  0.06450293213129044 , avg loss =  0.06725999435409903
    Finished training, iteration reached  20000  final loss =  0.06648300588130951  final avg =  0.0775011823300207
    saving snapshot to "lstm_demo3_2_snapshot_iter_20000.pt"



![png](output_24_1.png)



![png](output_24_2.png)


### Long term test


```python
test_net_iterative(lstmModel,3,1500)
```


![png](output_26_0.png)


Trained with sufficient long unrolled time window, the resulting network is capable of identifying frequency and phase of the sin() wave with high accuracy and generate a time-stable reproduction.
