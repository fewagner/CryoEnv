# mqtt info
broker = 'localhost'  # 172.19.4.159  # 'broker.hivemq.com'
port = 1883
username = 'fwagner'
password = '1234'

# constants
channel = 1
buffer_size = 10000
rseed = 1
adc_range = (-10., 10.)
dac_range = (0., 2.)
Ib_range = (0., 2.)
testpulse_interval = 2.
env_steps = 300
inference_steps = 30
record_length = 16384

# agent hyperpars
batch_size = 16
omega = 1e-2
penalty = 1.
gradient_steps = 100
lr = 1e-3
gamma = .6
learning_starts = 0
tpa_queue = [1]
pileup_prob = 0.
xi = 1e2
tau = 10

# paths
path_test = ''
path_models = path_test + 'models/'
path_buffer = path_test + 'data/'

# messages from control
subscribe_channel_msg = {
    'topic': 'ccscresst/subscription/set', 
    'keys': ['SubscribeToChannel'],
}

set_pars_msg = {
    'topic': 'ccscresst/control/set', 
    'keys': ['ChannelID', 'DAC', 'BiasCurrent'],
}

# messages from ccs
subscribe_acknowledge_msg = {
    'topic': 'ccscresst/subscription/ack', 
    'keys': ['SubscribeToChannel'],
}

trigger_msg = {
    'topic': 'ccscresst/trigger/samples', 
    'keys': ['ChannelID', 'TPA', 'LBaseline', 'PulseHeight', 'RMS', 'DAC', 'BiasCurrent', 'Samples'],
}

acknowledge_msg = {
    'topic': 'ccscresst/control/ack', 
    'keys': ['ChannelID', 'DAC', 'BiasCurrent'],
}

# action space
# DAC, BiasCurrent

# observation space
# PulseHeight, RMS, BiasCurrent, DAC
