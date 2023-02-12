# mqtt info
broker = 'localhost'  # '172.19.4.159'  # 'broker.hivemq.com'
port = 1883
username = 'fwagner'
password = '1234'

# constants
channel = 6
buffer_size = 500
rseed = 2
adc_range = (-10., 10.)
dac_range = (0., 5.)
Ib_range = (.5, 5.)
testpulse_interval = 10.
env_steps = 200
steps_per_episode = 40
inference_steps = 40
record_length = 16384

# agent hyperpars
batch_size = 16
omega = 0.
update_factor = 0.005
penalty = 0.
gradient_steps = 20
lr = 3e-4
gamma = .99
learning_starts = 0
tpa_queue = [1]
pileup_prob = 0.
xi = 1e2
tau = 45
log_reward = False
tpa_in_state = True
sweep = True
load = False

# paths
path_test = 'firstv1_2/'
path_models = path_test + 'models/'
path_buffer = path_test + 'data/'

path_load = 'firstv1_1/'

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
