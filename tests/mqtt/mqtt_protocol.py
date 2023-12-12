# mqtt info
broker = 'localhost'  # '172.19.4.159'  # 'broker.hivemq.com'
port = 10401  # 1883
username = 'fwagner'
password = '1234'

# constants
channel = 2
buffer_size = 1300
rseed = 2
adc_range = (-1., 1.)
dac_range = (0., 5.)
Ib_range = (.5, 5.)
testpulse_interval = 10.
env_steps = 1200
steps_per_episode = 60
inference_steps = 40
record_length = 16384

# agent hyperpars
batch_size = 16
omega = 0.01
update_factor = 0.005
penalty = 0.
gradient_steps = 20
lr = 3e-4
gamma = .9
learning_starts = 0
tpa_queue = [0.1, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
pileup_prob = 0.
xi = 1e2
tau = 45
log_reward = False
tpa_in_state = True
sweep = True
load = True

# paths
path_test = 'runs/firstv1_45/'  # lngs: '/datastor2/mqtt_tests/firstv1_38/'
path_models = path_test + 'models/'
path_buffer = path_test + 'data/'

path_load = 'runs/firstv1_39/'  # lngs: '/datastor2/mqtt_tests/firstv1_37/'

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
